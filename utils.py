from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


def _create_optimizer(self, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(
        optimizer_parameters,
        lr=self.args.learning_rate,
        eps=self.args.adam_eps,
        betas=(self.args.adam_b1, self.args.adam_b2),
        correct_bias=self.args.adam_correct_bias,
    )

    def _create_scheduler(self, optimizer):
        warmup_steps = int(self.num_train_steps * self.args.warmup_proportion)
        if self.args.lr_schedule == "warmup_linear":
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)
        if self.args.lr_schedule == "warmup_constant":
            return get_constant_schedule_with_warmup(optimizer, warmup_steps)

        raise RuntimeError("Unsupported scheduler: " + self.args.lr_schedule)