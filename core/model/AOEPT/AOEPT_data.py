from core.model.Base.base_data import Food101_Dataset, HateMemes_Dataset, MMIMDB_Dataset
from core.model.ViLT.ViLT_data import ViLT_Collator, ViLT_Dataset


class AOEPT_Dataset(ViLT_Dataset):
    def __init__(self, split: str, **kargs):
        super().__init__(split, **kargs)
        pass


class AOEPT_Collator(ViLT_Collator):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class MMIMDB_AOEPT_Dataset(AOEPT_Dataset, MMIMDB_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class MMIMDB_AOEPT_Collator(AOEPT_Collator):
    pass


class Food101_AOEPT_Dataset(AOEPT_Dataset, Food101_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class Food101_AOEPT_Collator(AOEPT_Collator):
    pass


class HateMemes_AOEPT_Dataset(AOEPT_Dataset, HateMemes_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class HateMemes_AOEPT_Collator(AOEPT_Collator):
    pass
