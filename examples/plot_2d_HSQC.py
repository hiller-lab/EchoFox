from echofox.echofoxplot import echofoxplot as fp



# simple 2d plot
def example_1():
    fp.plot2d(
        [
        ('Spectrum 1', fp.config.ECHOFOX_EXAMPLE_DATA_DIR+'/15N_HSQC_pipe/spectrum1.ft2','pipe' )
        ],
        min_sino = 4,
    )


    fp.savefig(f'{fp.config.ECHOFOX_EXAMPLE_DATA_DIR+'/15N_HSQC_pipe/spectrum1.ft2'[:-4]}.pdf')
    fp.show()

def example_2():
    fp.plot2d(
        [
        ('Spectrum 1', fp.config.ECHOFOX_EXAMPLE_DATA_DIR+'/13C_HSQC_pipe/spectrum1.ft2','pipe' )
        ],
        min_sino = 8,
    )


    fp.savefig(f'{fp.config.ECHOFOX_EXAMPLE_DATA_DIR+'/13C_HSQC_pipe/spectrum1.ft2'[:-4]}.pdf')
    fp.show()




if __name__ == '__main__':
    example_2()
