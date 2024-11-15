from separator import utils
from separator.engine import MusicVocalsSeparatorEngine

if __name__ == "__main__":
    args = utils.parse_args()
    MusicVocalsSeparatorEngine(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        _use_mdx_c=args.mdx_c,
        _use_vr_de_echo=args.de_echo,
        _use_vr_de_noise=args.de_noise,
        _keep_cache=args.keep_cache,
    ).run()
