from echofox.echofoxplot.config import config as plot_config
from echofox.nmr.config import config as nmr_config

print(plot_config)
print(nmr_config)


# Change the default save format
print(plot_config.default_save_format)
plot_config.default_save_format = "pdf"
print(plot_config.default_save_format)


# Echofox directories
print(plot_config.ECHOFOX_MAIN_DIR)
print(plot_config.ECHOFOX_EXAMPLE_DATA_DIR)
