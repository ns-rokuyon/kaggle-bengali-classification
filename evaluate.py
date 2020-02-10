import torch
import tqdm
import model as M
import preprocessing as P
import dataset as D
from train import evaluate
from data import get_logger
from pathlib import Path


data_dir = Path('data')
#model_file = 'workspace/model/run-4-seresnext50-cosanl-128px-bs96/run-4-seresnext50-cosanl-128px-bs96_best.model'
model_file = 'workspace/model/run-6-seresnext50-cosanl-128px-bs96-ohem-cutmix/run-6-seresnext50-cosanl-128px-bs96-ohem-cutmix_best.model'
device = torch.device('cuda:0')


def main():
    logger = get_logger(__name__)
    model = M.BengaliSEResNeXt50(pretrained=False)
    weight = torch.load(model_file, map_location='cpu')
    model.load_state_dict(weight)
    model = model.to(device)

    transformer = P.create_testing_transformer_v1(input_size=(128, 128))
    _, dataset = D.bengali_dataset(data_dir,
                                   fold_id=0,
                                   val_transformer=transformer,
                                   logger=logger)

    loader = torch.utils.data.dataloader.DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    score = evaluate(model, loader, device=device)
    logger.info(f'Score={score}')



if __name__ == '__main__':
    main()