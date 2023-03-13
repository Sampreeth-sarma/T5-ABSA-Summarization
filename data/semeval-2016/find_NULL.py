import xml.etree.ElementTree as ET
import pandas as pd

for idx, xml_tree in enumerate([ET.parse("Restaurants_Train_Final.xml"),
                                ET.parse("Restaurants_Test.xml")]):
    print(f"*********{'training set ' if idx == 0 else 'test set'} *****************")
    root = xml_tree.getroot()
    all_opinions = []
    all_text = []
    for node in root.iter('Review'):
        for sen in node.iter('sentence'):
            if sen.find("text").text is None or sen.find("Opinions") is None: continue
            sent_inp_text = ""
            op_dict = {}
            for elem in sen.iter():
                if elem.tag == "text":
                    sent_inp_text = elem.text
                if elem.tag == "Opinion":
                    pol = elem.attrib["polarity"]
                    asp = " ".join(elem.attrib["category"].split("#"))
                    tgt = elem.attrib["target"]
                    asp_pol = f"{asp} {pol}"
                    if asp_pol not in op_dict:
                        op_dict[asp_pol] = {"has_NULL": False, "target": []}
                    if tgt == "NULL":
                        op_dict[asp_pol]["has_NULL"] = True
                    else:
                        op_dict[asp_pol]["target"].append(tgt)
            all_opinions.append(op_dict)
            all_text.append(sent_inp_text)

    count = 0
    for op, txt in zip(all_opinions, all_text):
        for asp_pol in op.keys():
            if op[asp_pol]["has_NULL"] and len(op[asp_pol]["target"]) > 0:
                count += 1
                print(f"{count}:\n{txt}\n{asp_pol}: {op[asp_pol]['target']}\n\n")

    text_gen_data = pd.read_csv("test_TASD.csv")

