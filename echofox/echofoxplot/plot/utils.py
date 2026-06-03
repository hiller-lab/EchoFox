from matplotlib.lines import Line2D

from echofox.nmr.spectrum import NmrSpectrum


def _add_label(ax, label_list, label_kwargs_list):
    for i, label in enumerate(label_list):
        label_kwargs = label_kwargs_list[i]

        # add default kwargs for label to label_kwargs
        label_kwargs.setdefault("transform", ax.transAxes)
        label_kwargs.setdefault("fontsize", 12)
        label_kwargs.setdefault("verticalalignment", "top")
        label_kwargs.setdefault("horizontalalignment", "left")
        label_kwargs.setdefault("label_position", (0.02, 0.98))

        if label_kwargs["label_position"] in ["top-left", "top-right"]:
            padding = 5

            # Convert padding from points to axes coordinates
            pad_x = padding / ax.figure.dpi  # Convert pixels to figure-relative units
            pad_y = padding / ax.figure.dpi

            # Compute the top-left corner position (adjusted for padding)
            ax_x_min, ax_x_max = ax.get_xlim()
            ax_y_min, ax_y_max = ax.get_ylim()

            if label_kwargs["label_position"] == "top-left":
                # Adjust position to ensure the text fits inside the axes
                text_x = ax_x_min + pad_x * (ax_x_max - ax_x_min)
                text_y = ax_y_max - pad_y * (ax_y_max - ax_y_min)
                label_horizontalalignment = "left"

            elif label_kwargs["label_position"] == "top-right":
                # Adjust position to ensure the text fits inside the axes
                text_x = ax_x_max - pad_x * (ax_x_max - ax_x_min)
                text_y = ax_y_max - pad_y * (ax_y_max - ax_y_min)
                label_horizontalalignment = "right"
            else:
                text_x, text_y = label_kwargs["label_position"]
                label_horizontalalignment = "left"

            label_kwargs["horizontalalignment"] = label_horizontalalignment
            label_kwargs.pop("transform")

        else:
            text_x, text_y = label_kwargs["label_position"]

        # remove non-matplotlib kwargs from dict
        label_kwargs.pop("label_position")

        ax.text(text_x, text_y, label, **label_kwargs)


def _add_legend(ax, legend_kwargs):
    custom_lines = []
    custom_labels = []
    for i, spectrum in enumerate(ax.spectra):
        if isinstance(spectrum[0], NmrSpectrum) and spectrum[0].ndim == 1:
            dimensionality = 1
        elif isinstance(spectrum[0], NmrSpectrum) and spectrum[0].ndim == 2:
            dimensionality = 2
        else:
            raise ValueError("Spectrum must be an NmrSpectrum object.")

        name = spectrum[0].name
        spectrum_properties = spectrum[1]
        if dimensionality == 2:
            if "contour_color_positive" in spectrum_properties.keys():
                color = spectrum_properties["contour_color_positive"]
            elif "contour_color_negative" in spectrum_properties.keys():
                color = spectrum_properties["contour_color_negative"]
            else:
                color = ax.cmap_positive[i % len(ax.cmap_positive)]

        elif dimensionality == 1:
            if "spectrum_color" in spectrum_properties.keys():
                color = spectrum_properties["spectrum_color"]
            else:
                color = ax.color1d[i % len(ax.color1d)]
        else:
            color = "black"

        custom_lines.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
            )
        )
        custom_labels.append(name)

    if not "loc" in legend_kwargs.keys():
        legend_kwargs["loc"] = "upper left"
    if not "frameon" in legend_kwargs.keys():
        legend_kwargs["frameon"] = False
    if not "handlelength" in legend_kwargs.keys():
        legend_kwargs["handlelength"] = 1

    legend = ax.legend(custom_lines, custom_labels, **legend_kwargs)
