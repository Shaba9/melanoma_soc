from argparse import ArgumentParser

if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("--exp",choices=["baseline","augmented"])
    args=p.parse_args()
    print(f"Experiment scaffold: {args.exp}")
