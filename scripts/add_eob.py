import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_name",required=True, type = str, help="Predicted file name")
args = parser.parse_args()

with open(args.file_name, "r") as F:
    data = F.readlines()
add_eob_after = "System : "
data = [add_eob_after.join(i.split(add_eob_after)[:-1]+ [" <EOB> " + i.split(add_eob_after)[-1]]) for i in data]
print(data[0:3])

with open(args.file_name, "w") as F:
    F.writelines(data)
