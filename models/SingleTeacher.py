import tensorflow as tf
from models.ops import count_params, create_optimizer
from models.layers import layer_norm, conv1d, cq_attention, cq_concat, matching_loss, localizing_loss, ans_predictor
from models.lossfunc import *
from models.modules import word_embs, char_embs, add_pos_embs, conv_block, conditioned_predictor, dual_attn_block


class SingleTeacher:
    def __init__(self, configs, graph, word_vectors=None):
        self.configs = configs
        graph = graph if graph is not None else tf.Graph()
        with graph.as_default():
            self.global_step = tf.compat.v1.train.create_global_step()
            self._add_placeholders()
            self._build_model(word_vectors=word_vectors)
            # if configs.mode == 'train':
            #     print('\x1b[1;33m' + 'Total trainable parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)
            # else:
            #     print('\x1b[1;33m' + 'Total parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)

    def _add_placeholders(self):
        self.video_inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.configs.model.vdim],
                                           name='video_inputs')
        self.video_seq_len = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='video_seq_len')
        self.word_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')
        self.char_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None, None], name='char_ids')
        self.y1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='start_indexes')
        self.y2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='end_indexes')
        self.match_labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name='match_labels')
        # hyper-parameters
        self.drop_rate = tf.compat.v1.placeholder_with_default(input=0.0, shape=[], name='dropout_rate')
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32, name='learning_rate')

    def _build_model(self, word_vectors):
        # create mask for both visual and textual features
        v_mask = tf.sequence_mask(lengths=self.video_seq_len, maxlen=tf.reduce_max(input_tensor=self.video_seq_len), dtype=tf.int32) #(16, 64)
        q_mask = tf.cast(tf.cast(self.word_ids, dtype=tf.bool), dtype=tf.int32) #(16, 11)

        # text_encoder:: generate query features
        word_emb = word_embs(self.word_ids, dim=self.configs.model.word_dim, drop_rate=self.drop_rate, finetune=False,
                             reuse=False, vectors=word_vectors) 
        char_emb = char_embs(self.char_ids, char_size=self.configs.num_chars, dim=self.configs.model.char_dim, reuse=False,
                             kernels=[1, 2, 3, 4], filters=[10, 20, 30, 40], drop_rate=self.drop_rate, padding='VALID',
                             activation=tf.nn.relu)
        word_emb = tf.concat([word_emb, char_emb], axis=-1) 
        qfeats = conv1d(word_emb, dim=self.configs.model.dim, use_bias=True, reuse=False, name='query_conv1d')
        qfeats = layer_norm(qfeats, reuse=False, name='q_layer_norm')


        # generate video features
        vfeats = tf.nn.dropout(self.video_inputs, rate=self.drop_rate)
        vfeats = conv1d(vfeats, dim=self.configs.model.dim, use_bias=True, reuse=False, name='video_conv1d')
        vfeats = layer_norm(vfeats, reuse=False, name='v_layer_norm') 


        # add positional embedding and convolutional block
        vfeats = add_pos_embs(vfeats, max_pos_len=self.configs.model.max_vlen, reuse=False, name='pos_emb')
        vfeats = conv_block(vfeats, kernel_size=7, dim=self.configs.model.dim, num_layers=4, drop_rate=self.drop_rate,
                            activation=tf.nn.relu, reuse=False, name='conv_block')
        qfeats = add_pos_embs(qfeats, max_pos_len=self.configs.model.max_vlen, reuse=True, name='pos_emb')
        qfeats = conv_block(qfeats, kernel_size=7, dim=self.configs.model.dim, num_layers=4, drop_rate=self.drop_rate,
                            activation=tf.nn.relu, reuse=True, name='conv_block')
        t0_vfeats, t0_qfeats = vfeats, qfeats

        # teacher attention block
        for li in range(self.configs.model.attn_layer):
            vfeats_ = dual_attn_block(t0_vfeats, t0_qfeats, dim=self.configs.model.dim, num_heads=self.configs.model.num_heads,
                                      from_mask=v_mask, to_mask=q_mask, use_bias=True, drop_rate=self.drop_rate, activation=None, reuse=False, name='t0_d_attn_%d' % li)
            qfeats_ = dual_attn_block(t0_qfeats, t0_vfeats, dim=self.configs.model.dim, num_heads=self.configs.model.num_heads,
                                      from_mask=q_mask, to_mask=v_mask, use_bias=True, drop_rate=self.drop_rate, activation=None, reuse=True, name='t0_d_attn_%d' % li)
            t0_vfeats, t0_qfeats = vfeats_, qfeats_

        t0_q2v_feats, _ = cq_attention(t0_vfeats, t0_qfeats, mask1=v_mask, mask2=q_mask, drop_rate=self.drop_rate, reuse=False, name='t0_q2v_attn')
        t0_v2q_feats, _ = cq_attention(t0_qfeats, t0_vfeats, mask1=q_mask, mask2=v_mask, drop_rate=self.drop_rate, reuse=False, name='t0_v2q_attn')
        t0_fuse_feats = cq_concat(t0_q2v_feats, t0_v2q_feats, pool_mask=q_mask, reuse=False, name='t0_cq_cat')

        t0_match_loss, t0_match_scores = matching_loss(t0_fuse_feats, self.match_labels, label_size=4, mask=v_mask,
                                                      gumbel=not self.configs.loss.no_gumbel, tau=self.configs.loss.tau, reuse=False, name="t0_match_loss")
        t0_label_embs = tf.compat.v1.get_variable(name='t0_label_emb', shape=[4, self.configs.model.dim], dtype=tf.float32, trainable=True, initializer=tf.compat.v1.orthogonal_initializer())
        t0_ortho_constraint = tf.multiply(tf.matmul(t0_label_embs, t0_label_embs, transpose_b=True), 1.0 - tf.eye(4, dtype=tf.float32))
        t0_ortho_constraint = tf.norm(tensor=t0_ortho_constraint, ord=2)
        t0_match_loss += t0_ortho_constraint


        t0_soft_label_embs = tf.matmul(t0_match_scores, tf.tile(tf.expand_dims(t0_label_embs, axis=0), multiples=[tf.shape(t0_match_scores)[0], 1, 1]))
        t0_outputs = (t0_fuse_feats + t0_soft_label_embs) * tf.cast(tf.expand_dims(v_mask, axis=-1), dtype=tf.float32)
        t0_start_logits,t0_end_logits = conditioned_predictor(t0_outputs, dim=self.configs.model.dim, reuse=False, mask=v_mask,
                                                         num_heads=self.configs.model.num_heads, drop_rate=self.drop_rate,
                                                         attn_drop=self.drop_rate, max_pos_len=self.configs.model.max_vlen,
                                                         activation=tf.nn.relu, name="t0_predictor")
        t0_loc_loss = localizing_loss(t0_start_logits, t0_end_logits, self.y1, self.y2, v_mask)

        # fuse features
        q2v_feats, _ = cq_attention(vfeats, qfeats, mask1=v_mask, mask2=q_mask, drop_rate=self.drop_rate, reuse=False, name='studen_q2v_attn')
        v2q_feats, _ = cq_attention(qfeats, vfeats, mask1=q_mask, mask2=v_mask, drop_rate=self.drop_rate, reuse=False, name='studen_v2q_attn')
        fuse_feats = cq_concat(q2v_feats, v2q_feats, pool_mask=q_mask, reuse=False, name='studen_cq_cat')


        # compute matching loss and matching score
        self.match_loss, self.match_scores = matching_loss(fuse_feats, self.match_labels, label_size=4, mask=v_mask,
                                                      gumbel=not self.configs.loss.no_gumbel, tau=self.configs.loss.tau, reuse=False, name="match_loss")
        label_embs = tf.compat.v1.get_variable(name='student_label_emb', shape=[4, self.configs.model.dim], dtype=tf.float32,
                                     trainable=True, initializer=tf.compat.v1.orthogonal_initializer())
        ortho_constraint = tf.multiply(tf.matmul(label_embs, label_embs, transpose_b=True), 1.0 - tf.eye(4, dtype=tf.float32))
        ortho_constraint = tf.norm(tensor=ortho_constraint, ord=2)  # compute l2 norm as loss
        self.match_loss += ortho_constraint


        
        soft_label_embs = tf.matmul(self.match_scores, tf.tile(tf.expand_dims(label_embs, axis=0),  multiples=[tf.shape(input=self.match_scores)[0], 1, 1]))
        outputs = (fuse_feats + soft_label_embs) * tf.cast(tf.expand_dims(v_mask, axis=-1), dtype=tf.float32)
        # compute start and end logits
        self.start_logits, self.end_logits = conditioned_predictor(outputs, dim=self.configs.model.dim, reuse=False, mask=v_mask,
                                                         num_heads=self.configs.model.num_heads, drop_rate=self.drop_rate,
                                                         attn_drop=self.drop_rate, max_pos_len=self.configs.model.max_vlen,
                                                         activation=tf.nn.relu, name="predictor")

        # compute localization loss
        # std = tf.math.reduce_variance(self.start_logits) + tf.math.reduce_variance(self.end_logits)
        # self.loc_loss = self.loc_loss / std + std
        self.start_index, self.end_index = ans_predictor(self.start_logits, self.end_logits, v_mask, "slow")

        self.loc_loss = localizing_loss(self.start_logits, self.end_logits, self.y1, self.y2, v_mask)

        inter_kdfunc = eval(self.configs.loss.inter_kdfunc)
        inter_loss_0 = inter_kdfunc(outputs, t0_outputs)
        inter_loss_1 = inter_kdfunc(fuse_feats, t0_fuse_feats)

        ss_loss = self.loc_loss + self.configs.loss.match_lambda * self.match_loss
        t0_loss = t0_loc_loss + self.configs.loss.match_lambda * t0_match_loss
        inter_loss = self.configs.loss.inter_cof_0 * inter_loss_0 + self.configs.loss.inter_cof_1 * inter_loss_1

        self.loss = ss_loss + t0_loss + inter_loss 
                    
        # create optimizer
        self.train_op = create_optimizer(self.loss, self.lr, clip_norm=self.configs.train.clip_norm)
