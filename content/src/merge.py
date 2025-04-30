files = ["./content/src/bmp.py", "./content/src/wav.py", "./content/src/convertor.py", "./content/src/main.py"]


out = "./content/src/bckp.py"


with open(out, "w") as out:
    for file in files:
        with open(file, "r") as file:
            out.write(file.read())
            out.write("\n\n")
