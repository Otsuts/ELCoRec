import os
import datetime


def write_log(w, args):
    file_name = (
        args.log
        + "/"
        + args.dataset
        + "/"
        + args.model
        + "/"
        + datetime.date.today().strftime("%m%d")
        + f"_LR{args.lr}.log"
    )
    if not os.path.exists(args.log + "/" + args.dataset + "/" + args.model + "/"):
        os.makedirs(args.log + "/" + args.dataset + "/" + args.model + "/")
    t0 = datetime.datetime.now().strftime("%H:%M:%S")
    info = "{} : {}".format(t0, w)
    print(info)
    if not args.test_mode:
        with open(file_name, "a") as f:
            f.write(info + "\n")
