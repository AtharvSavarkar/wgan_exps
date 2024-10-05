import os


def make_dirs_for_wgan_generate(budget, algo):
    try:
        os.mkdir(f'generated_imgs_{algo}_{budget}')
    except FileExistsError:
        pass

    try:
        os.mkdir(f'generated_imgs_{algo}_{budget}/mel')
    except FileExistsError:
        pass

    try:
        os.mkdir(f'generated_imgs_{algo}_{budget}/nv')
    except FileExistsError:
        pass
