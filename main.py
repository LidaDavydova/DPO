import argparse
import dpo_train
import test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training and saving to {args.model_path}")
    else:
        print(f"Loading model from {args.model_path} for evaluation")

if __name__ == "__main__":
    main()