import warnings
from sklearn.exceptions import InconsistentVersionWarning

import rl_data_gen.generate as gen

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)



def main():
    model = gen.load_model()
    anchors = gen.extract_anchor_pairs()
    rand_anchors, trend_anchors = gen.split_anchors(anchors)

    down, up = gen.classify_trend(trend_anchors, model)
    print(f"Trend: {len(trend_anchors)} episodes (down={down}, up={up})")
    print(f"Random-walk: {len(rand_anchors)} episodes")

    train_dir, val_dir = gen.prepare_output_dirs()

    episodes = []
    episodes += gen.generate_episodes(trend_anchors, model, mode="trend")
    episodes += gen.generate_episodes(rand_anchors, model, mode="random_walk")

    print(f"Total episodes: {len(episodes)}")
    gen.write_episodes(episodes, train_dir, val_dir)
    print("Generation complete.")

if __name__ == "__main__":
    main()