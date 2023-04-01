from functools import partial

import numpy as np
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import get_2d_sincos_pos_embed
from .dino import MultiCropWrapper, cosine_scheduler

from .vision_transformer import PatchEmbed, DINOHead, Block


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        diag_att=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.diag_att = diag_att

    def forward(self, x):
        q, k, v = x
        B, N, C = q.shape

        q = (
            self.to_q(q)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B h N C//h
        k = (
            self.to_k(k)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B h N C//h
        v = (
            self.to_v(v)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B h N C//h

        # B h N C//h x B h C//h N => B h N N
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # B h N N
        attn = self.attn_drop(attn)  # B h N N

        if self.diag_att:
            attn = torch.diagonal(attn, dim1=-2, dim2=-1)  # B h N
            attn = attn.unsqueeze(-1)  # B h N 1

        # B h N N x B h N C//h => B h N C//h => B N h C//h => B N C
        if self.diag_att:
            x = (attn * v).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Backbone(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_chans=3,
        embed_dim=512,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        # x = x[:, 1:, :]

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        has_decoder=True,
        aux_cls=False,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        att_loss=False,
        diag_att=False,
        # DINO params,
        enable_dino=False,
        out_dim=128,
        local_crops_number=8,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        epochs=200,
    ):
        super().__init__()

        self.backbone = Backbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        self.num_patches = self.backbone.patch_embed.num_patches

        self.att = CrossAttention(
            dim=embed_dim,
            num_heads=1,
            qkv_bias=True,
            diag_att=diag_att,
        )
        self.diag_att = diag_att

        self.aux_cls = aux_cls
        if self.aux_cls:
            self.fc = nn.Linear(embed_dim, 1)

        self.enable_dino = enable_dino
        if self.enable_dino:
            # student DINO
            self.student = MultiCropWrapper(
                self.backbone,
                DINOHead(
                    embed_dim,
                    out_dim,
                    use_bn=False,
                    norm_last_layer=True,
                ),
            )

            teacher = Backbone(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )

            self.teacher = MultiCropWrapper(
                teacher,
                DINOHead(
                    embed_dim,
                    out_dim,
                    use_bn=False,
                    norm_last_layer=True,
                ),
            )

            self.dino_loss_fn = DINOLoss(
                out_dim,
                local_crops_number + 2,
                warmup_teacher_temp,
                teacher_temp,
                warmup_teacher_temp_epochs,
                epochs,
            )

            self.teacher.load_state_dict(self.student.state_dict())

            self.momentum_schedule = None
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------

        self.has_decoder = has_decoder
        if self.has_decoder:
            # MAE decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.backbone.num_patches + 1, decoder_embed_dim),
                requires_grad=False,
            )  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
            )  # decoder to patch
            # --------------------------------------------------------------------------

            self.norm_pix_loss = norm_pix_loss

        self.att_loss = att_loss

        self.initialize_weights()

    def get_last_selfattention(self, images):
        return self.backbone.get_last_selfattention(images)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.pos_embed.shape[-1],
        #     int(self.backbone.num_patches ** 0.5),
        #     cls_token=True,
        # )
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.has_decoder:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.backbone.num_patches ** 0.5),
                cls_token=True,
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
            )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_momentum_schedule(self, momentum_teacher, epochs, num_iters):
        if self.enable_dino:
            self.momentum_schedule = cosine_scheduler(
                momentum_teacher, 1, epochs, num_iters
            )

    def update_teacher_ema(self, it):
        if self.enable_dino:
            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(
                    self.student.parameters(), self.teacher.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.backbone.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.backbone.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_replace(self, x, x_ref, mask_ratio):
        assert x.shape == x_ref.shape
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)

        # replace
        # generate the binary mask: 0 is keep, 1 is replace
        ids_replace = ids_shuffle[:, len_keep:]
        mask = torch.zeros([N, L], device=x.device)

        for i in range(N):
            x[i][ids_replace[i]] = x_ref[i][ids_replace[i]]
            mask[i][ids_replace[i]] = 1

        return x, mask

    def forward_encoder_(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_encoder(self, x, x_ref):
        x = self.backbone(x)
        x_ref = self.backbone(x_ref)

        x = x[:, 1:]
        x_ref = x_ref[:, 1:]

        x, att = self.att([x, x_ref, x])

        return x, att, None

    @torch.no_grad()
    def extract_features(self, x):
        x = self.backbone(x)
        x = x[:, 1:]
        return x

    @torch.no_grad()
    def get_attmap(self, x, x_ref):
        features, attentions = self.att([x, x_ref, x])  # B h N N
        h = attentions.shape[2]
        nh = attentions.shape[1]
        att_h = att_w = int(h ** 0.5)
        diag_attentions = [attentions[:, :, i, i].unsqueeze(-1) for i in range(h)]
        diag_attentions = torch.cat(diag_attentions, axis=-1)
        diag_attentions = diag_attentions.reshape(-1, nh, att_h, att_w)

        patch_size = self.backbone.patch_embed.patch_size[0]
        diag_attentions = nn.functional.interpolate(
            diag_attentions,
            scale_factor=patch_size,
            mode="nearest",
        )

        diag_attentions *= 1 / (diag_attentions.max())
        diag_attentions = 1 - diag_attentions
        return features, diag_attentions

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def reconstruction_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        return loss.mean()

    def classification_loss(self, pred, mask):
        # import pdb; pdb.set_trace()
        return F.binary_cross_entropy_with_logits(pred, mask)

    def diag_att_loss(self, diag, masks):
        return F.smooth_l1_loss(diag, masks)

    def segmentation_loss(self, pred, mask):
        return self.segmentation_criterion(pred, mask)

    def dino_loss(self, teacher_output, student_output, epoch):
        loss = self.dino_loss_fn(student_output, teacher_output, epoch)
        return loss

    def forward_dino(self, images, epoch):
        # import pdb; pdb.set_trace()
        teacher_output = self.teacher(images[:2])
        student_output = self.student(images)
        loss = self.dino_loss(teacher_output, student_output, epoch)
        return loss

    def forward(self, batch, epoch=0):
        rep_frame = batch["rep_frame"]
        ref_frame = batch["ref_frame"]
        cur_frame = batch["cur_frame"]
        masks = batch["masks"]

        latent, att, _ = self.forward_encoder(rep_frame, ref_frame)

        loss = 0
        ret_dict = {}

        if self.has_decoder:
            pred = self.forward_decoder(latent)  # [N, L, p*p*3]
            rec_loss = self.reconstruction_loss(cur_frame, pred)
            ret_dict["pred"] = pred
            loss += rec_loss
            ret_dict["rec_loss"] = rec_loss
            # loss.append(rec_loss)

        if self.training and self.enable_dino:
            # DINO training
            ref_frame_dino = batch["ref_frame_dino"]
            dino_loss = self.forward_dino(ref_frame_dino, epoch)
            loss += dino_loss

            # ret_dict["loss"] = loss
            ret_dict["dino_loss"] = dino_loss

        ret_dict["loss"] = loss

        # loss = sum(loss) / len(loss)

        ret_dict.update({"loss": loss, "att": att})
        return ret_dict


def mae_vit_small_patch8(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8,
        embed_dim=512,
        depth=4,
        num_heads=4,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
