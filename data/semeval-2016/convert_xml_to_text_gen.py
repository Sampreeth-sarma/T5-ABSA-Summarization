import xml.etree.ElementTree as ET
import pandas as pd

for idx, xml_tree in enumerate([ET.parse("ABSA16FR_Restaurants_Train-withcontent.xml"),
                                ET.parse("ABSA16FR_Restaurants_Gold-withcontent.xml")]):
    print(f"*********{'training set ' if idx == 0 else 'test set'} *****************")
    root = xml_tree.getroot()
    inp_text, target_text, ids = [], [], []
    yes_no = []
    sentence = []
    aspect_sentiment = []
    all_aspect_sentiment = ["restaurant general positive", "restaurant general negative", "restaurant general neutral",
                            "service general positive", "service general negative", "service general neutral",
                            "food quality positive", "food quality negative", "food quality neutral",
                            "food style_options positive", "food style_options negative", "food style_options neutral",
                            "drinks style_options positive", "drinks style_options negative",
                            "drinks style_options neutral", "drinks prices positive", "drinks prices negative",
                            "drinks prices neutral", "restaurant prices positive", "restaurant prices negative",
                            "restaurant prices neutral", "restaurant miscellaneous positive",
                            "restaurant miscellaneous negative", "restaurant miscellaneous neutral",
                            "ambience general positive", "ambience general negative", "ambience general neutral",
                            "food prices positive", "food prices negative", "food prices neutral",
                            "location general positive", "location general negative", "location general neutral",
                            "drinks quality positive", "drinks quality negative", "drinks quality neutral"]
    asp_pol_map = {val: idx for idx, val in enumerate(all_aspect_sentiment)}
    ner_tags = []
    for node in root.iter('Review'):
        for sen in node.iter('sentence'):
            if sen.find("text").text is None or sen.find("Opinions") is None: continue
            sent_inp_text = ""
            sent_target_text = []
            sent_asp_pol = all_aspect_sentiment
            sent_yes_no = [0] * 36
            sent_ner_tags = []
            id = ""
            for elem in sen.iter():
                if elem.tag == "sentence":
                    id += elem.attrib["id"]
                if elem.tag == "text":
                    sent_inp_text += elem.text
                    print(sent_inp_text)
                    sent_ner_tags.extend([["O"] * len(sent_inp_text.split()) for _ in range(36)])
                if elem.tag == "Opinion":
                    polarity = elem.attrib['polarity'].lower()
                    category = " ".join(elem.attrib['category'].lower().split("#"))
                    target = elem.attrib['target']
                    from_idx = int(elem.attrib["from"])
                    to_idx = int(elem.attrib["to"])
                    num_words_before_target = sent_inp_text[: from_idx].count(" ")

                    category_polarity = category + " " + polarity
                    if category_polarity in asp_pol_map:
                        sent_yes_no[asp_pol_map[category_polarity]] = 1
                        sent_target_text.append(
                            f"{polarity} opinion on {category} for {target}")
                        for offset, each_word in enumerate(target.split()):
                            if each_word != "NULL":
                                sent_ner_tags[asp_pol_map[category_polarity]][num_words_before_target + offset] = "T"

            if len(sent_target_text) > 0:
                sent_target_text = "The review expressed " + " , ".join(sent_target_text)
                inp_text.append(sent_inp_text)
                target_text.append(sent_target_text)
                ids.extend([id] * 36)
                yes_no.extend(sent_yes_no)
                aspect_sentiment.extend(sent_asp_pol)
                sentence.extend([sent_inp_text] * 36)
                ner_tags.extend([" ".join(each) for each in sent_ner_tags])

    df = pd.DataFrame(ids, columns=["sentence_id"])
    df["yes_no"] = yes_no
    df["aspect_sentiment"] = aspect_sentiment
    df["sentence"] = sentence
    df["ner_tags"] = ner_tags
    prefix = ["TASD" for _ in range(len(inp_text))]
    t5_input_df = pd.DataFrame(prefix, columns=["prefix"])
    t5_input_df["input_text"] = inp_text
    t5_input_df["target_text"] = target_text

    if idx == 0:
        df.to_csv("train_TAS_french.tsv", sep="\t", header=True, index=False)
        t5_input_df.to_csv("train_TASD_french.csv", header=True, index=False)
    else:
        df.to_csv("test_TAS_french.tsv", sep="\t", header=True, index=False)
        t5_input_df.to_csv("test_TASD_french.csv", header=True, index=False)
