from lib.algos.sigcwgan import SigCWGANConfig
from lib.augmentations import get_standard_augmentation, SignatureConfig, Scale, Concat, Cumsum, AddLags, LeadLag

SIGCWGAN_CONFIGS = dict(
    HIGH_FREQ_AAPL=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    HIGH_FREQ_GOOG=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
     HIGH_FREQ_NVDA=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    HIGH_FREQ_ORCL=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
)
