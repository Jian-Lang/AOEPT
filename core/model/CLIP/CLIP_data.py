from ..Base.base_data import Food101_Dataset, HateMemes_Dataset, MMIMDB_Dataset
from ..Unified.Unified_data import Unified_Collator, Unified_Dataset


class CLIP_Dataset(Unified_Dataset):
    def __init__(self, split: str, **kargs):
        super().__init__(split, **kargs)


class CLIP_Collator(Unified_Collator):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class MMIMDB_CLIP_Dataset(CLIP_Dataset, MMIMDB_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class MMIMDB_CLIP_Collator(CLIP_Collator):
    pass


class Food101_CLIP_Dataset(CLIP_Dataset, Food101_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class Food101_CLIP_Collator(CLIP_Collator):
    pass


class HateMemes_CLIP_Dataset(CLIP_Dataset, HateMemes_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)


class HateMemes_CLIP_Collator(CLIP_Collator):
    pass
