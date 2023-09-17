import xml.etree.ElementTree as ET
import sys
import os
import json

if len(sys.argv) != 2:
    print("The applications needs a path to a xml file to convert")
    exit(1)

path = sys.argv[1]
paths = []
directory = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if not name.endswith("xml"):
                continue
            print(os.path.join(root, name))
            paths.append(os.path.join(root, name))
else:
    paths.append(path)

for f in paths:
    interfaces = []
    tree = ET.parse(f)
    root = tree.getroot()

    print(f)
    try:
        for element in root.find("elements").findall("element"):
            d = {}
            for child in element.iter():
                if child.tag == "height":
                    d["apertureDiameter"] = float(child.text)
                if child.tag == "thickness":
                    d["thickness"] = float(child.text)
                if child.tag == "radius":
                    d["radius"] = float(child.text)
                if child.tag == "refractiveIndex":
                    d["ior"] = float(child.text)
                if child.tag == "abbeNumber":
                    d["abbeNumber"] = float(child.text)
                if child.tag == "coatingLambda":
                    d["coatingLambda"] = float(child.text)
                if child.tag == "coatingIor":
                    d["coatingIor"] = float(child.text)
            interfaces.append(d)
    except Exception:
        pass

    filename = os.path.basename(f)
    filename = os.path.splitext(filename)[0]
    with open(os.path.join(directory, f"{filename}.json"), "w") as f:
        json.dump(interfaces, f)
