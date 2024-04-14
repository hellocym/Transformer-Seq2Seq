class PeptideDataset(Dataset):
    def __init__(self, files, split='train', transform=None):
        # 初始为空的列表，用于存储各个分割的数据
        train_data_list = []
        val_data_list = []
        test_data_list = []

        # 对每个文件单独进行划分
        for file in files:
            # 读取文件
            data = pd.read_csv(file, header=None, skiprows=1)
            # 取第一列和第二列数据
            data = data.iloc[:, :2]

            # 划分数据
            train_temp, temp_data = train_test_split(
                data, test_size=0.3, random_state=42)
            val_temp, test_temp = train_test_split(
                temp_data, test_size=1/3, random_state=42)

            # 追加到相应的列表中
            train_data_list.append(train_temp)
            val_data_list.append(val_temp)
            test_data_list.append(test_temp)

        # 合并来自所有文件的数据
        if split == 'train':
            self.data = pd.concat(train_data_list, ignore_index=True)
        elif split == 'val':
            self.data = pd.concat(val_data_list, ignore_index=True)
        elif split == 'test':
            self.data = pd.concat(test_data_list, ignore_index=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        column1 = row[0]
        column2 = row[1]
        if self.transform:
            column1 = self.transform(column1)
            column2 = self.transform(column2)
        return (column1, column2)
