import argparse
import dpo_train
import test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--checkpoint_dir', required=False)
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training")
        dpo_train.main()
    else:
        print(f"Loading model from {args.checkpoint_dir} for evaluation")
        test.main(args.checkpoint_dir)

if __name__ == "__main__":
    main()