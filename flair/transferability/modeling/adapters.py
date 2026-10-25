"""
This script contains adapters for fast adaptation of
FLAIR modelo to downstream tasks/domains.

In particular, these adapters work over the vision and text
embeddings. Also, this code contains a Wrapper for zero-shot
classification

Implemented adapters:
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import copy as copy_module   # not patched by the benchmark (it replaces copy.deepcopy only)
import os
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.modeling.dictionary import definitions
from flair.pretraining.data.transforms import augmentations_pretraining

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
The first section contains only-vision adapters (i.e. linear probing)
"""


class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        # Set model and number of targets
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.num_targets = len(targets)
        # Augmentation for training and for test-time augmentation
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20

    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()

        epoch_iterator = tqdm(
            data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True
        )

        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision encoder
                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]

        if self.fta:
            transforms = augmentations_pretraining

        # Extract features and labels from generator
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        # Perform logistic regression
        self.train(X, Y)

    def train(self, X, Y):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return

    def predict(self, loader, transforms=None):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return


class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0,
                                             class_weight="balanced")

    def train(self, X, Y):

        # Train classifier
        self.classifier.fit(X, Y)

        # Set Linear Probe classifier into FLAIR model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    def predict(self, loader, transforms=None):
        self.model.eval()

        # Set transforms on test-time augmentation
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(
            loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            refs, preds = [], []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # Forward
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))
                    score = torch.concat(preds_tta, -1).mean(-1)
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)
                # Activation for prediction
                if score.shape[-1] == 1:  # Binary case
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)
                else:  # Multi-class case
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
This section contains multimodal (vision-language) adapters.
"""


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Compute text prototypes
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()),
                                                                                domain_knowledge=domain_knowledge)


class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    def fit(self, loaders, transforms=None):
        """
        No training in zero-shot prediction
        """
        return

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds


class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.ratio = 0.2

        # Set adapter
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    # Compute residual CLIP-Adapter
                    X = self.residual_adapter(X)
                    # Compute similarity
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                # Compute residual CLIP-Adapter
                X = self.residual_adapter(X)
                # Compute similarity
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        # TRAINING
        epochs, lr, bs = 40, 0.001, 1

        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)
        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)

                # Compute residual CLIP-Adapter
                X_batch = self.residual_adapter(X_batch)

                # Compute logits
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    def residual_adapter(self, X):
        # Compute residual CLIP-Adapter
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)
        return X


class KnowledgeConditionedResidual(torch.nn.Module):
    """
    Residual vision adapter conditioned on the expert-text class embeddings.

    Given an image embedding x (B x D) and class text embeddings T (C x D):
      1) knowledge summary  t_hat = softmax(tau * x T^T) T   (what the expert text says about
         the classes this image looks like; equals zero-shot FLAIR's posterior over classes)
      2) residual           r = MLP([x ; t_hat])
      3) output             x' = normalize(x + ratio * r)
    The last MLP layer is zero-initialised, so before training x' == x and the adapted model is
    exactly zero-shot FLAIR. This is the only difference with CLIP-Adapter, whose residual is a
    function of x alone.
    """

    def __init__(self, c_in, reduction=4, ratio=0.2):
        super().__init__()
        self.ratio = ratio
        self.fc1 = torch.nn.Linear(2 * c_in, c_in // reduction, bias=False)
        self.act = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(c_in // reduction, c_in, bias=False)
        torch.nn.init.zeros_(self.fc2.weight)

    def forward(self, x, text_embeds, logit_scale):
        # Knowledge summary from the (frozen) expert-text prototypes
        attn = torch.softmax(logit_scale * (x @ text_embeds.t()), dim=-1)
        t_hat = attn @ text_embeds
        # Knowledge-conditioned residual
        r = self.fc2(self.act(self.fc1(torch.cat([x, t_hat], dim=-1))))
        x = x + self.ratio * r
        return x / x.norm(dim=-1, keepdim=True)


class KnowledgeAdapter(ClipAdapter):
    """
    Knowledge-conditioned adapter (project contribution). Same training recipe, hyper-parameters
    and inference path as ClipAdapter (see ClipAdapter.train / predict); only the residual changes.
    """

    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        # Replace CLIP-Adapter's residual MLP by the knowledge-conditioned one
        self.adapter = KnowledgeConditionedResidual(self.c_in, reduction=self.reduction,
                                                    ratio=self.ratio).to(device)

    def residual_adapter(self, X):
        X = X.to(torch.float32)
        text_embeds = self.text_embeds.to(device).to(torch.float32)
        return self.adapter(X, text_embeds, self.model.logit_scale.exp())


class KnowledgeCacheAdapter(LanguageAdapterWrapper):
    """
    Knowledge-conditioned adapter v2 (description-level expert knowledge + support cache).

    FLAIR's zero-shot classifier averages the expert descriptions of a class into ONE prototype,
    which blurs lesion-specific cues (e.g. "venous beading" vs "neovascularization"). Here each
    class keeps its K descriptions e_ck and the image chooses how much each one matters:

        z_ck(x)  = s * <x, e_ck>
        w_ck(x)  = softmax_k( gamma * z_ck(x) + u_ck )          (gamma, u_ck learnable)
        S_c(x)   = sum_k w_ck(x) * z_ck(x)                      (knowledge score of class c)
        logits   = S_c(x) + alpha * Tip-cache_c(x)              (training-free cache, as Tip-Adapter)

    With gamma = 0 and u = 0 the weights are uniform, so S_c(x) = s * <x, mean_k e_ck> is exactly
    zero-shot FLAIR, and logits are exactly Tip-Adapter's: training can only move away from that.
    Only gamma and u (1 + C*K numbers) are trained, by cross-entropy on the support set with the
    cache switched OFF during training (a support image would otherwise match itself in the cache).
    use_cache=False gives the text-only ablation.
    """

    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, use_cache=True):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        # Same constants as TipAdapter (inference path)
        self.scale = 100.0
        self.beta = 5
        self.alpha = 1
        self.use_cache = use_cache
        self.cache_keys = None
        self.cache_values = None

        # Per-description expert-text embeddings, padded to K: (C x K x D) + validity mask (C x K)
        self.desc_embeds, self.desc_mask = self.compute_description_embeddings(list(targets.keys()),
                                                                               domain_knowledge)
        n_classes, n_desc, _ = self.desc_embeds.shape
        self.gamma = torch.nn.Parameter(torch.zeros(1, device=device))
        self.u = torch.nn.Parameter(torch.zeros(n_classes, n_desc, device=device))

    def compute_description_embeddings(self, categories, domain_knowledge):
        # Same description lists / prompt as FLAIRModel.compute_text_embeddings (which already ran)
        lists = []
        for c in categories:
            if domain_knowledge and c in definitions:
                descs = list(definitions[c])
                if c not in descs and "myopic maculopathy" not in c:
                    descs.append(c)
            else:
                descs = [c]
            lists.append(descs)

        k_max = max(len(d) for d in lists)
        dim = self.text_embeds.shape[-1]
        embeds = torch.zeros(len(lists), k_max, dim, device=device)
        mask = torch.zeros(len(lists), k_max, dtype=torch.bool, device=device)
        for i, descs in enumerate(lists):
            with torch.no_grad():
                prompts = [self.model.caption.replace("[CLS]", d) for d in descs]
                tokens = self.model.text_model.tokenizer(prompts, truncation=True, padding=True,
                                                         return_tensors='pt')
                input_ids = tokens["input_ids"].to(device).to(torch.long)
                attention_mask = tokens["attention_mask"].to(device).to(torch.long)
                emb = self.model.text_model(input_ids, attention_mask)
            embeds[i, :len(descs)] = emb.to(torch.float32)
            mask[i, :len(descs)] = True
        return embeds, mask

    def zero_shot_logits(self, X):
        return self.scale * (X @ self.text_embeds.to(device).to(torch.float32).t())

    def text_logits(self, X):
        # z: B x C x K similarities to every expert description
        z = self.scale * torch.einsum('bd,ckd->bck', X, self.desc_embeds)
        weights = self.gamma * z + self.u.unsqueeze(0)
        weights = weights.masked_fill(~self.desc_mask.unsqueeze(0), float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        return (weights * z).sum(-1)

    def cache_logits(self, X):
        affinity = X @ self.cache_keys
        return torch.exp((-1) * (self.beta - self.beta * affinity)) @ self.cache_values

    def logits(self, X):
        out = self.text_logits(X)
        if self.use_cache and self.cache_keys is not None:
            out = out + self.alpha * self.cache_logits(X)
        return out

    def train(self, X, Y):
        X = torch.tensor(X).to(torch.float32)
        Y = torch.tensor(Y).to(torch.long)

        # Training-free cache from the support set (Tip-Adapter)
        self.cache_keys = X.t().to(device)
        self.cache_values = torch.nn.functional.one_hot(Y, num_classes=self.num_targets).to(torch.float32).to(device)

        # Learn description weighting (cache OFF so a support image cannot match itself).
        # Learning rate chosen a priori (few scalars, few steps); NOT tuned on test data.
        epochs, lr = 40, 0.01
        optimizer = torch.optim.AdamW([self.gamma, self.u], lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in np.random.permutation(X.shape[0]):
                x_batch = X[i_sample, :].unsqueeze(0).to(device)
                target = Y[i_sample].unsqueeze(0).to(device)

                loss = torch.nn.functional.cross_entropy(self.text_logits(x_batch), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item() / X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")
        print('gamma=%2.4f' % self.gamma.item(), end="\n")

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                with torch.no_grad():
                    score_i = self.logits(torch.tensor(X).to(device).to(torch.float32))
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            with torch.no_grad():
                score = self.logits(torch.tensor(X).to(device).to(torch.float32))
                if os.environ.get("KC_DEBUG"):
                    self.print_debug(torch.tensor(X).to(device).to(torch.float32))

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds

    def print_debug(self, X):
        # Diagnostic (KC_DEBUG=1): how big are the class-to-class logit gaps of each term?
        # What matters for the argmax is the spread ACROSS classes, so we report its mean std.
        def spread(t):
            return t.std(dim=-1).mean().item()
        zs = self.zero_shot_logits(X)
        tl = self.text_logits(X)
        cl = self.alpha * self.cache_logits(X) if self.cache_keys is not None else torch.zeros_like(tl)
        agree_zs = (tl.argmax(-1) == zs.argmax(-1)).float().mean().item()
        agree_total = (self.logits(X).argmax(-1) == tl.argmax(-1)).float().mean().item()
        print("DEBUG gamma=%.4f  u(mean|u|)=%.4f  max-weight(mean)=%.3f" % (
            self.gamma.item(), self.u.abs().mean().item(),
            self._mean_max_weight(X)))
        print("DEBUG class-spread: zero-shot=%.3f  learned-text=%.3f  cache=%.3f" % (spread(zs), spread(tl), spread(cl)))
        print("DEBUG argmax agreement: learned-text vs zero-shot=%.3f  final vs learned-text=%.3f" % (agree_zs, agree_total))

    def _mean_max_weight(self, X):
        z = self.scale * torch.einsum('bd,ckd->bck', X, self.desc_embeds)
        w = self.gamma * z + self.u.unsqueeze(0)
        w = torch.softmax(w.masked_fill(~self.desc_mask.unsqueeze(0), float('-inf')), dim=-1)
        multi = self.desc_mask.sum(1) > 1
        return w.max(-1).values[:, multi].mean().item() if multi.any() else 1.0


class KnowledgeProtoAdapter(KnowledgeCacheAdapter):
    """
    Knowledge-initialised prototype adapter (v3). Builds on KnowledgeCacheAdapter:

        logits_c(x) = S_c(x) + <x, delta_c> + b_c + alpha * Tip-cache_c(x)

    S_c(x): description-attention score over FLAIR's expert descriptions (zero-shot FLAIR when gamma=u=0).
    delta, b: learnable prototype residual / class bias, initialised at 0 and L2-penalised (lam), so the model
              starts at zero-shot FLAIR and can move the decision boundary (v2 could not).
    cache:    training-free support-set cache, alpha/beta chosen on a validation split by the benchmark.
    use_attn=False freezes the description attention (gamma=u=0); use_res=False freezes delta and b.
    """

    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False,
                 use_attn=True, use_res=True, lam=0.1, alpha=1.0, beta=5, steps=100, lr=0.05):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta, use_cache=True)
        self.use_attn, self.use_res = use_attn, use_res
        self.lam, self.steps, self.lr = lam, steps, lr
        self.alpha, self.beta = alpha, beta

        n_classes, _, dim = self.desc_embeds.shape
        self.delta = torch.nn.Parameter(torch.zeros(n_classes, dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(n_classes, device=device))

    def text_logits(self, X):
        return super().text_logits(X) + X @ self.delta.t() + self.bias

    def train(self, X, Y):
        X = torch.tensor(X).to(torch.float32).to(device)
        Y = torch.tensor(Y).to(torch.long).to(device)

        # Training-free cache from the support set (used only at inference)
        self.cache_keys = X.t()
        self.cache_values = torch.nn.functional.one_hot(Y, num_classes=self.num_targets).to(torch.float32)

        params = []
        if self.use_attn:
            params += [self.gamma, self.u]
        if self.use_res:
            params += [self.delta, self.bias]
        if len(params) == 0:
            return

        # Full-batch training on the cached support features, cache OFF (a support image would match itself).
        # lr / steps are fixed a priori; only lam (and the cache alpha/beta) are tuned, on validation data.
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0, eps=1e-4)
        for _ in range(self.steps):
            loss = torch.nn.functional.cross_entropy(self.text_logits(X), Y) + self.lam * (self.delta ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class KnowledgeHeadAdapter(KnowledgeCacheAdapter):
    """
    Knowledge-initialised prototype head + support cache (v4, "ours2"). No description attention.

        logits_c(x) = s * <x, t_c + delta_c> + alpha * Tip-cache_c(x)

    t_c: FLAIR's expert-description prototype (mean of the class's expert-text embeddings), so with
         delta = 0 and alpha = 0 the model is exactly zero-shot FLAIR.
    delta: learnable per-class residual added to the text prototype, trained by cross-entropy on the support
         set with an L2 anchor lam * ||delta||^2 toward the text prototype. It is multiplied by the same logit
         scale s as the text term (v3's residual was not, so its penalty froze it at ~0). Strong anchor ->
         zero-shot behaviour (few shots); weak anchor -> linear-probe behaviour (many shots).
    cache: training-free Tip-style cache over the support features; alpha, beta and lam are chosen by the
         benchmark (support-set cross-validation + validation split), never on test data.
    use_head=False -> text prototype + cache only. fresh(lam) clones the (frozen) text side cheaply.
    """

    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False,
                 lam=1.0, alpha=1.0, beta=5, steps=100, lr=0.002, use_head=True):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta, use_cache=True)
        self.lam, self.alpha, self.beta = lam, alpha, beta
        self.steps, self.lr, self.use_head = steps, lr, use_head
        self._new_head()

    def _new_head(self):
        n_classes, _, dim = self.desc_embeds.shape
        self.delta = torch.nn.Parameter(torch.zeros(n_classes, dim, device=device))
        self.cache_keys, self.cache_values = None, None

    def fresh(self, lam, use_head=None):
        clone = copy_module.copy(self)          # shares the frozen model / text embeddings
        clone.lam = lam
        if use_head is not None:
            clone.use_head = use_head
        clone._new_head()
        return clone

    def text_logits(self, X):
        out = super().text_logits(X)            # gamma = u = 0 (never trained) -> mean expert prototype
        if self.use_head:
            out = out + self.scale * (X @ self.delta.t())
        return out

    def train(self, X, Y):
        X = torch.tensor(X).to(torch.float32).to(device)
        Y = torch.tensor(Y).to(torch.long).to(device)

        self.cache_keys = X.t()
        self.cache_values = torch.nn.functional.one_hot(Y, num_classes=self.num_targets).to(torch.float32)

        if not self.use_head:
            return
        # Full-batch training on cached support features, cache OFF (a support image would match itself).
        # lr / steps fixed a priori; lam is selected on held-out data by the benchmark.
        optimizer = torch.optim.AdamW([self.delta], lr=self.lr, weight_decay=0.0, eps=1e-4)
        for _ in range(self.steps):
            loss = torch.nn.functional.cross_entropy(self.text_logits(X), Y) + self.lam * (self.delta ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.train_tip = train

        # Init cache values
        self.cache_keys = []
        self.cache_values = []
        self.adapter_layer = []

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)

        if self.train_tip:

            # TRAINING
            epochs, lr, bs = 40, 0.001, 1

            # Enable the cached keys to be learnable
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)

            indexes = np.arange(0, self.cache_keys.shape[1])
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)

                    # Zero-shot CLIP
                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())

                    # Tip-Adapter
                    affinity = adapter_layer(image)
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()

                    tip_logits = clip_logits + cache_logits * self.alpha

                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    def adapter(self, X):
        # Zero-shot CLIP
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))

        # Tip-Adapter
        if not self.train_tip:
            affinity = X @ self.cache_keys
        else:
            affinity = self.adapter_layer(X)

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
        logits = clip_logits + cache_logits * self.alpha

        return logits