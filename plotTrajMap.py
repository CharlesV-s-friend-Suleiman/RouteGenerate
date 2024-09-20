import pandas as pd
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg

df = pd.read_csv('data/artificial_traj3.csv')
fig, axs = plt.subplots(1, 3, figsize=(30, 10))

color_map = {'GG': 'r', 'GSD': 'b', 'TG': 'g'}
bg_img_gg = mpimg.imread('figur/GG_realworld.jpg')
bg_img_gsd = mpimg.imread('figur/GSD_realworld.jpg')
bg_img_tg = mpimg.imread('figur/TG_realworld.jpg')

traj_ids = df['traj_id'].unique()

for mode, ax, bg_img in zip(['GG', 'GSD', 'TG'], axs, [bg_img_gg, bg_img_gsd, bg_img_tg]):
    ax.imshow(bg_img, extent=[0, 364, 0, 326], aspect='auto')
    for traj_id in traj_ids:
        df_traj = df[df['traj_id'] == traj_id]
        if df_traj['mode'].values[0] == mode:
            ax.plot(df_traj['locx'], df_traj['locy'], color=color_map[mode], label=mode)
    ax.set_title(f'Trajectories of mode {mode}')
    ax.set_xlabel('locx')
    ax.set_ylabel('locy')
    ax.set_aspect('equal')
    ax.set_xlim(0, 364)
    ax.set_ylim(0, 326)

plt.show()