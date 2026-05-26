from ..Base.base_data import Food101_Dataset, HateMemes_Dataset, MMIMDB_Dataset
from ..Unified.Unified_data import Unified_Collator, Unified_Dataset


class ViLT_Dataset(Unified_Dataset):
    def __init__(self, split: str, **kargs):
        super().__init__(split, **kargs)


class ViLT_Collator(Unified_Collator):
    def __init__(self, statis: str = None, **kargs):
        super().__init__(**kargs)
        self.statis = statis
        self.collect_token = (statis == "collect_token")


class MMIMDB_ViLT_Dataset(ViLT_Dataset, MMIMDB_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class MMIMDB_ViLT_Collator(ViLT_Collator):
    pass


class Food101_ViLT_Dataset(ViLT_Dataset, Food101_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class Food101_ViLT_Collator(ViLT_Collator):
    pass


class HateMemes_ViLT_Dataset(ViLT_Dataset, HateMemes_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class HateMemes_ViLT_Collator(ViLT_Collator):
    pass
