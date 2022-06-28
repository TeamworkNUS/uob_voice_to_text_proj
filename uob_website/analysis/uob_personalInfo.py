import re
import os
import pandas as pd
from text2digits import text2digits
from analysis.uob_init import(
    kyc_product_list
)

def personalInfoDetector(sttDf):
    if os.path.exists(path=kyc_product_list):
        print('Extract KYC words to list...')
        template = pd.read_csv(kyc_product_list)
        kycWordsList = template['words'][template['category'] == 'kyc'].tolist()

    t2d = text2digits.Text2Digits()
    sttDf['text'] = [t2d.convert(x) for x in sttDf['text']]
    # kyc detector
    kycList = [ ]
    for i in range(len(sttDf)):
        if any(x in sttDf['text'][i].split() for x in kycWordsList):
            kycList.append('Yes')
        # if re.findall(r'\d+\b', sttDf['text'][i]) and (any(x in sttDf['text'][i].split() for x in kycWordsList) or any(x in sttDf['text'][i-1].split() for x in kycWordsList)):
        #     kycList.append('Yes')
        else:
            kycList.append(' ')
    sttDf['is_kyc'] = kycList
    
    # pii detector
    piiList=[ ]
    list_of_sentences = sttDf['text'].values
    for i in range(len(list_of_sentences)):
        if str(list_of_sentences[i]):
            text = str(list_of_sentences[i])
            parsed_text = PiiRegex(text)
            if parsed_text.any_match():
                piiList.append('Yes')
            else:
                piiList.append(' ')
        else:
            piiList.append(' ')
    sttDf['is_pii'] = piiList
    return sttDf



date = re.compile(
    u"(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}",
    re.IGNORECASE,
)
time = re.compile(u"\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?", re.IGNORECASE)
email = re.compile(
    u"([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)",
    re.IGNORECASE,
)

credit_card = re.compile(u"((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])")

street_address = re.compile(
    u"[\s](?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)",
    re.IGNORECASE,
)
zip_code = re.compile(r"[0-9]{6}$")
po_box = re.compile(r"P\.? ?O\.? Box \d+", re.IGNORECASE)

postcodes = re.compile("([gG][iI][rR] {0,}0[aA]{2})|((([a-pr-uwyzA-PR-UWYZ][a-hk-yA-HK-Y]?[0-9][0-9]?)|(([a-pr-uwyzA-PR-UWYZ][0-9][a-hjkstuwA-HJKSTUW])|([a-pr-uwyzA-PR-UWYZ][a-hk-yA-HK-Y][0-9][abehmnprv-yABEHMNPRV-Y]))) {0,}[0-9][abd-hjlnp-uw-zABD-HJLNP-UW-Z]{2})")
sghandphones = re.compile(r"[8,9][0-9]{7}$")
sglandingphone = re.compile(r"[6][0-9]{7}$")
sgvoip = re.compile(r"[3][0-9]{7}$")

regexes = {
    "dates": date,
    "times": time,
    "emails": email,
    "credit_cards": credit_card,
    "street_addresses": street_address,
    "zip_codes": zip_code,
    "po_boxes": po_box,
    "postcodes": postcodes,
    "sgphones": sghandphones,
    "sglandingphone": sglandingphone,
    "sgvoip": sgvoip
}


class regex:
    def __init__(self, obj, regex):
        self.obj = obj
        self.regex = regex

    def __call__(self, *args):
        def regex_method(text=None):
            return [x for x
                    in self.regex.findall(text or self.obj.text)]

        return regex_method


class PiiRegex(object):
    def __init__(self, text=""):
        self.text = text

        # Build class attributes of callables.
        for k, v in regexes.items():
            setattr(self, k, regex(self, v)(self))

        if text:
            for key in regexes.keys():
                method = getattr(self, key)
                setattr(self, key, method())

    def any_match(self, text=""):
        """Scan through all available matches and try to match.
        """
        if text:
            self.text = text

            # Regenerate class attribute callables.
            for k, v in regexes.items():
                setattr(self, k, regex(self, v)(self))
            for key in regexes.keys():
                method = getattr(self, key)
                setattr(self, key, method())

        matches = []
        for match in regexes.keys():
            # If we've got a result, add it to matches.
            if getattr(self, match):
                matches.append(match)

        return matches if matches else False

