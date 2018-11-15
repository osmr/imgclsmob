import pandas as pd


def get_validation_cls(working_dir_path):
    df = pd.read_csv("../../../imgclsmob_data/oi4bb/validation-annotations-bbox.csv")
    df2 = df.assign(Square=(df.XMax - df.XMin) * (df.YMax - df.YMin))
    df2 = df2[["ImageID", "LabelName", "Square"]]
    df2 = df2.loc[df2.groupby(["ImageID"])["Square"].idxmax()]
    df2 = df2[["ImageID", "LabelName"]]
    df2.to_csv("../../../imgclsmob_data/oi4bb/validation-cls.csv", index=False)
    pass


def main():

    pass


if __name__ == '__main__':
    main()
