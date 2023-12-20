from opencompass.models import MixtralSparseWemixSphinx

# Please follow the instruction in https://github.com/open-compass/MixtralKit
# to download the model weights and install the requirements


models = [
    dict(
        abbr="mixtral-sparse-wemix-sphinx",
        type=MixtralSparseWemixSphinx,
        # path="/mnt/petrelfs/share_data/gaopeng/mixtral-8x7b-32kseqlen/ori",
        # tokenizer_path="/mnt/petrelfs/share_data/gaopeng/mixtral-8x7b-32kseqlen/ori/tokenizer.model",
        # dialog_ultrachat200kWizardcode_mistral_lr0.000005
        # path="/mnt/petrelfs/share_data/gaopeng/dialog_select1_mistralSparseEns/",
        path="/mnt/petrelfs/share_data/gaopeng/dialog_select1_mistralSparseEns_iter209663",
        tokenizer_path="/mnt/petrelfs/huangsiyuan/mixtral/tokenizer.model",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        num_gpus=2,
        wemix_ratio=0.2,
        run_cfg=dict(num_gpus=2, num_procs=2),
    ),
]