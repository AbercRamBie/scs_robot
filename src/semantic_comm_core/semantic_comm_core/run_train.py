import argparse
from semantic_comm_core.trainer import train
from semantic_comm_core.config import Config


def _parse_snr_values(raw_values: str) -> list[float]:
    return [float(v.strip()) for v in raw_values.split(',') if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train semantic communication model across SNR values.'
    )
    parser.add_argument(
        '--snr-values',
        type=str,
        default='-10,-5,0,5,10,15,20',
        help='Comma-separated SNR values in dB.'
    )
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--bottleneck-dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel-type', type=str, default='rayleigh')
    parser.add_argument(
        '--run-name-prefix',
        type=str,
        default='rayleigh_snr_sweep',
        help='Prefix for run names. Final name is <prefix>_<snr>dB.'
    )

    args = parser.parse_args()
    snr_values = _parse_snr_values(args.snr_values)

    for snr in snr_values:
        cfg = Config(
            snr_db_train=snr,
            beta=args.beta,
            bottleneck_dim=args.bottleneck_dim,
            epochs=args.epochs,
            lr=args.lr,
            channel_type=args.channel_type,
            run_name=f'{args.run_name_prefix}_{snr}dB'
        )
        print(f'\nTraining with {args.channel_type} channel at SNR = {snr} dB')
        train(cfg)


if __name__ == '__main__':
    main()


