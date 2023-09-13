import xml.etree.ElementTree as ET
import sys
import os
import json

if len(sys.argv) != 2:
    print("The applications needs a path to a xml file to convert")
    exit(1)

directory = os.path.dirname(os.path.abspath(__file__))
interfaces = []
path = sys.argv[1]
tree = ET.parse(path)
root = tree.getroot()
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

print(interfaces)

filename = os.path.basename(path)
filename = os.path.splitext(filename)[0]
with open(os.path.join(directory, f"{filename}.json"), "w") as f:
    json.dump(interfaces, f)
