from dataset.build_dataset import build_dataset
from readingcomprehension.models.luke import LukeForReadingComprehensionWithLoss

if __name__ == "__main__":
    input_file = "./dataset/luke-squad-train.mindrecord"
    dataset_train = build_dataset(input_file)

    netwithloss = LukeForReadingComprehensionWithLoss()