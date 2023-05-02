Search.setIndex({"docnames": ["README", "data_cleaning", "main", "model_building", "project-description"], "filenames": ["README.md", "data_cleaning.ipynb", "main.ipynb", "model_building.ipynb", "project-description.md"], "titles": ["Project Group 28: Airbnb Europe Dataset Analysis", "Data Preprocessing", "A Comprehensive Analysis Using Machine Learning Techniques", "Model Building and Evaluation", "Final Project: original data analysis"], "terms": {"thi": [0, 1, 2, 3, 4], "aim": 0, "analyz": [0, 2], "which": [0, 2, 4], "avail": 0, "kaggl": [0, 2], "we": [0, 1, 2, 3, 4], "explor": [0, 2, 4], "variou": [0, 2], "data": 0, "clean": [0, 1, 2], "preprocess": [0, 3], "model": [0, 1, 4], "techniqu": 0, "extract": [0, 3], "insight": [0, 2], "from": [0, 1, 2, 3, 4], "make": [0, 2, 3, 4], "predict": [0, 2, 3], "The": [0, 1, 2, 4], "s": [0, 1, 2, 4], "jupyterbook": 0, "can": [0, 1, 2, 4], "access": [0, 4], "here": [0, 1, 2, 3], "follow": [0, 2], "contain": [0, 2, 4], "raw": [0, 2], "process": [0, 2, 3], "file": [0, 1, 4], "figur": [0, 1, 2, 3, 4], "gener": [0, 2, 4], "plot": [0, 1, 2, 3], "result": [0, 1, 3, 4], "obtain": 0, "train": [0, 2, 3], "evalu": [0, 4], "util": [0, 2, 3, 4], "function": 0, "modul": 0, "us": [0, 3, 4], "throughout": 0, "main": 0, "ipynb": [0, 4], "notebook": 0, "provid": [0, 2, 4], "an": [0, 2, 3, 4], "overview": 0, "data_clean": 0, "step": [0, 2, 3, 4], "model_build": 0, "environ": [0, 4], "yml": [0, 4], "requir": [0, 4], "packag": [0, 3], "makefil": [0, 4], "build": [0, 1], "manag": 0, "other": [0, 2, 4], "task": 0, "clone": 0, "git": [0, 4], "ucb": 0, "stat": [0, 1], "159": [0, 4], "s23": 0, "group28": 0, "cd": 0, "creat": [0, 1, 2, 3, 4], "activ": 0, "aemf": 0, "mamba": 0, "env": [0, 3, 4], "f": [0, 1, 4], "name": [0, 1, 2, 3], "conda": [0, 3], "ipython": [0, 2], "kernel": 0, "python": [0, 4], "m": 0, "ipykernel": 0, "user": [0, 2, 4], "displai": [0, 2, 3, 4], "To": [0, 1, 2], "local": [0, 4], "run": [0, 3, 4], "html": 0, "up": [0, 4], "execut": [0, 4], "all": [0, 4], "under": [0, 4], "bsd": 0, "3": [0, 1, 2, 3], "claus": 0, "import": [1, 2, 3, 4], "numpi": [1, 3, 4], "np": [1, 3], "panda": [1, 2, 3], "pd": [1, 2, 3], "matplotlib": [1, 3], "pyplot": [1, 3], "plt": [1, 3], "seaborn": 1, "sn": 1, "scipi": 1, "os": 1, "df": [1, 2], "read_csv": [1, 2, 3], "aemf1": [1, 2], "csv": [1, 2, 3], "head": [1, 2], "10": [1, 2, 3, 4], "citi": [1, 2, 3], "price": [1, 2, 3], "dai": [1, 2, 3], "room": [1, 2, 3], "type": [1, 2, 3, 4], "share": [1, 2, 3, 4], "privat": [1, 2, 3], "person": [1, 2, 3, 4], "capac": [1, 2, 3], "superhost": [1, 2, 3], "multipl": [1, 2, 3], "busi": [1, 2, 3], "cleanli": [1, 2, 3], "rate": [1, 2, 3], "guest": [1, 2, 3], "satisfact": [1, 2, 3], "bedroom": [1, 2, 3], "center": [1, 2, 3], "km": [1, 2, 3], "metro": [1, 2, 3], "distanc": [1, 2, 3], "attract": [1, 2, 3], "index": [1, 2, 3], "normalis": [1, 2, 3], "restraunt": [1, 2, 3], "0": [1, 2, 3], "amsterdam": [1, 2], "194": [1, 2], "033698": [1, 2], "weekdai": [1, 2], "fals": [1, 2, 3], "true": [1, 2], "2": [1, 2, 3, 4], "1": [1, 2, 3], "93": [1, 2], "5": [1, 2, 3], "022964": [1, 2], "539380": [1, 2], "78": [1, 2], "690379": [1, 2], "4": [1, 2, 3], "166708": [1, 2], "98": [1, 2], "253896": [1, 2], "6": [1, 2], "846473": [1, 2], "344": [1, 2], "245776": [1, 2], "8": [1, 2, 3], "85": [1, 2], "488389": [1, 2], "239404": [1, 2], "631": [1, 2], "176378": [1, 2], "33": [1, 2], "421209": [1, 2], "837": [1, 2], "280757": [1, 2], "58": [1, 2], "342928": [1, 2], "264": [1, 2], "101422": [1, 2], "9": [1, 2], "87": [1, 2], "748312": [1, 2], "651621": [1, 2], "75": [1, 2], "275877": [1, 2], "985908": [1, 2], "95": [1, 2], "386955": [1, 2], "646700": [1, 2], "433": [1, 2], "529398": [1, 2], "90": [1, 2], "384862": [1, 2], "439876": [1, 2], "493": [1, 2], "272534": [1, 2], "26": [1, 2, 3], "119108": [1, 2], "875": [1, 2], "033098": [1, 2], "60": [1, 2], "973565": [1, 2], "485": [1, 2], "552926": [1, 2], "544738": [1, 2], "318693": [1, 2], "552": [1, 2], "830324": [1, 2], "29": [1, 2], "272733": [1, 2], "815": [1, 2], "305740": [1, 2], "56": [1, 2], "811677": [1, 2], "808567": 1, "100": [1, 4], "131420": 1, "904668": 1, "174": 1, "788957": 1, "255191": 1, "225": 1, "201662": 1, "15": 1, "692376": 1, "215": 1, "124317": 1, "94": 1, "881092": 1, "729747": 1, "200": 1, "167652": 1, "599010": 1, "242": 1, "765524": 1, "16": 1, "916251": 1, "7": [1, 2, 3], "2771": 1, "307384": 1, "entir": [1, 2], "home": [1, 2], "apt": [1, 2], "686807": 1, "458404": 1, "208": 1, "808109": 1, "11": 1, "056528": 1, "272": 1, "313823": 1, "18": 1, "975219": 1, "1001": 1, "804420": 1, "96": 1, "719141": 1, "196112": 1, "106": 1, "226456": 1, "624761": 1, "133": 1, "876202": 1, "328686": 1, "276": 1, "521454": 1, "88": 1, "142361": 1, "924404": 1, "206": [1, 2], "252862": 1, "921226": 1, "238": 1, "291258": 1, "604478": 1, "isna": 1, "sum": 1, "dtype": [1, 3], "int64": 1, "dataset": [1, 2, 3, 4], "ar": [1, 2, 4], "work": [1, 2, 4], "europ": [1, 2], "doesn": [1, 4], "t": [1, 4], "have": [1, 2, 4], "ani": [1, 2, 4], "miss": [1, 2], "na": 1, "valu": [1, 2, 3], "howev": 1, "should": [1, 4], "still": 1, "check": 1, "potenti": [1, 2, 4], "outlier": [1, 2], "could": [1, 2], "affect": [1, 2], "our": [1, 2, 4], "perform": [1, 2, 3], "first": [1, 2, 3], "let": 1, "histogram": [1, 2], "show": [1, 3], "frequenc": [1, 2], "distribut": [1, 2], "figsiz": [1, 3], "histplot": 1, "kde": 1, "titl": [1, 3, 4], "xlabel": [1, 3], "ylabel": 1, "savefig": [1, 3], "price_distribution_befor": 1, "png": [1, 2], "dpi": 1, "300": 1, "bbox_inch": [1, 3], "tight": [1, 3], "observ": [1, 2], "seem": [1, 3], "address": [1, 4], "issu": 1, "remov": [1, 2, 3], "base": [1, 2, 3], "interquartil": [1, 2], "rang": [1, 2, 4], "iqr": [1, 2], "method": [1, 2], "code": [1, 3], "oper": 1, "price_summari": 1, "describ": [1, 4], "print": 1, "to_csv": [1, 2, 3], "count": 1, "41714": 1, "000000": 1, "mean": [1, 2, 3], "260": 1, "094423": 1, "std": 1, "279": 1, "408493": 1, "min": 1, "34": 1, "779339": 1, "25": [1, 2], "144": [1, 2], "016085": 1, "50": 1, "203": 1, "819274": 1, "297": 1, "373358": 1, "max": 1, "18545": 1, "450285": 1, "float64": 1, "after": 1, "save": [1, 4], "filter": [1, 2], "q1": [1, 2], "quantil": [1, 2], "q3": [1, 2], "lower_bound": [1, 2], "upper_bound": [1, 2], "out": [1, 2], "filtered_data": [1, 2, 3], "price_distribut": 1, "With": 1, "now": 1, "readi": 1, "further": [1, 2, 3], "analysi": 1, "calcul": [1, 2, 3], "correl": [1, 2], "matrix": [1, 2], "corr_matrix": 1, "corr": 1, "heatmap": 1, "visual": [1, 2], "12": [1, 3], "annot": 1, "cmap": 1, "coolwarm": 1, "fmt": 1, "2f": 1, "linewidth": 1, "custom": 1, "featur": [1, 3], "xtick": 1, "rotat": 1, "45": 1, "ha": [1, 2, 4], "right": [1, 4], "tight_layout": 1, "feature_correlation_heatmap": [1, 2], "tmp": 1, "ipykernel_3674": 1, "1166516350": 1, "py": [1, 3], "futurewarn": [1, 3], "default": [1, 3], "numeric_onli": 1, "datafram": [1, 3, 4], "deprec": 1, "In": [1, 2], "futur": [1, 2], "version": [1, 3], "select": 1, "onli": 1, "valid": [1, 2, 3], "column": [1, 3], "specifi": [1, 3], "silenc": 1, "warn": [1, 3], "boxplot": [1, 2], "compar": [1, 2, 4], "x": [1, 3], "y": [1, 3], "comparison": 1, "price_comparison_by_room_typ": [1, 2], "value_count": 1, "25728": 1, "12781": 1, "314": 1, "entire_hom": 1, "private_room": 1, "shared_room": 1, "subplot": [1, 2], "fig": 1, "ax": 1, "scatterplot": 1, "alpha": 1, "color": 1, "ee7600": 1, "set_titl": 1, "vs": 1, "set_xlabel": 1, "set_ylabel": 1, "green": 1, "befor": 1, "price_vs_distance_from_city_center_by_room_typ": [1, 2], "city_stat": [1, 2], "groupbi": 1, "agg": 1, "median": [1, 2], "convert": [1, 2], "numer": [1, 2], "city_label": 1, "astyp": 1, "categori": 1, "cat": 1, "between": [1, 2], "city_price_corr": 1, "pearson": 1, "iloc": 1, "path": 1, "exist": [1, 3], "makedir": 1, "folder": 1, "10361768269037437": [1, 2], "354": [1, 2], "109178": [1, 2], "356": [1, 2], "197127": [1, 2], "athen": [1, 2], "354423": [1, 2], "127": [1, 2], "715417": [1, 2], "barcelona": [1, 2], "227": [1, 2], "110980": [1, 2], "196": [1, 2], "895292": [1, 2], "berlin": [1, 2], "211": [1, 2], "988194": [1, 2], "185": [1, 2], "566047": [1, 2], "budapest": [1, 2], "167": [1, 2], "491323": [1, 2], "152": [1, 2], "277107": [1, 2], "lisbon": [1, 2], "230": [1, 2], "253192": [1, 2], "223": [1, 2], "030019": [1, 2], "pari": [1, 2], "299": [1, 2], "216175": [1, 2], "287": [1, 2], "305434": [1, 2], "rome": [1, 2], "197": [1, 2], "229417": [1, 2], "182": [1, 2], "124237": [1, 2], "vienna": [1, 2], "221": [1, 2], "750960": [1, 2], "390389": [1, 2], "averag": [1, 2], "each": [1, 2, 3, 4], "city_pric": 1, "sort_valu": 1, "ascend": 1, "bar": [1, 2], "relationship": [1, 2], "barplot": 1, "average_price_by_c": [1, 2], "label": [1, 4], "x_var": 1, "y_var": 1, "loop": 1, "through": [1, 2], "variabl": [1, 2, 4], "separ": [1, 4], "i": [1, 4], "enumer": 1, "hue": 1, "_vs_": 1, "imag": 2, "pipeline_util": [2, 3], "create_pipelin": [2, 3], "jupyt": [2, 4], "book": 2, "dive": 2, "world": 2, "airbnb": 2, "becom": 2, "popular": 2, "platform": 2, "travel": 2, "find": [2, 3, 4], "accommod": 2, "offer": [2, 4], "wide": 2, "option": 2, "across": 2, "understand": 2, "factor": 2, "influenc": 2, "essenti": 2, "host": 2, "who": [2, 4], "want": [2, 4], "optim": [2, 3], "list": [2, 4], "well": [2, 4], "best": [2, 3, 4], "deal": [2, 4], "goal": [2, 4], "locat": 2, "amen": [2, 4], "more": [2, 4], "By": [2, 4], "do": [2, 4], "so": [2, 4], "hope": 2, "valuabl": 2, "both": 2, "inform": [2, 4], "decis": [2, 4], "european": 2, "includ": [2, 4], "found": 2, "weekend": 2, "restaur": 2, "preview": 2, "section": [2, 4], "origin": [2, 3], "had": 2, "upon": 2, "decid": 2, "them": [2, 3, 4], "determin": 2, "lower": [2, 4], "upper": 2, "bound": 2, "better": 2, "gain": 2, "identifi": [2, 4], "filenam": 2, "differ": [2, 4], "allow": 2, "reveal": 2, "built": 2, "primari": 2, "object": [2, 3], "most": [2, 3, 4], "effect": 2, "purpos": [2, 4], "achiev": 2, "split": [2, 3], "test": 2, "set": [2, 3, 4], "help": [2, 4], "tune": [2, 3], "hyperparamet": [2, 3], "emploi": 2, "final": 2, "chosen": 2, "project": 2, "random": [2, 3, 4], "forest": [2, 3], "lasso": [2, 3], "regress": 2, "ridg": [2, 3], "These": 2, "combin": [2, 3], "imput": [2, 3], "simpl": [2, 4], "k": 2, "nearest": 2, "neighbor": 2, "handl": 2, "total": [2, 3, 4], "six": [2, 3], "pair": 2, "repres": 2, "pipelin": [2, 3], "simple_imput": [2, 3], "rf": [2, 3], "simpleimput": [2, 3], "strategi": [2, 3], "most_frequ": [2, 3], "randomforestregressor": [2, 3], "min_samples_leaf": [2, 3], "knn_imput": [2, 3], "knnimput": [2, 3], "next": 2, "grid": [2, 3], "search": [2, 3], "metric": [2, 3], "r": [2, 3], "squar": [2, 3], "error": [2, 3], "mse": [2, 3], "absolut": [2, 3], "mae": [2, 3], "summari": [2, 3, 4], "748426": [2, 3], "3500": [2, 3], "114743": [2, 3], "41": [2, 3], "321086": [2, 3], "578790": [2, 3], "6102": [2, 3], "000292": [2, 3], "57": [2, 3], "295679": [2, 3], "578681": [2, 3], "5051": [2, 3], "988150": [2, 3], "52": [2, 3], "254937": [2, 3], "749541": [2, 3], "were": 2, "knn": 2, "regressor": 2, "yield": 2, "among": 2, "indic": [2, 4], "good": 2, "balanc": 2, "accuraci": 2, "results_df": [2, 3], "74413": [2, 3], "2966": [2, 3], "950719": [2, 3], "37": [2, 3], "245278": [2, 3], "signific": [2, 4], "contribut": [2, 4], "underli": 2, "pattern": 2, "guid": 2, "overal": 2, "serv": 2, "reliabl": [2, 4], "tool": 2, "target": [2, 4], "given": 2, "feature_importance_plot": [2, 3], "take": [2, 4], "you": [2, 3, 4], "how": [2, 4], "leverag": 2, "experi": 2, "satisfactori": 2, "kei": [2, 3, 4], "benefici": 2, "when": 2, "appropri": 2, "As": 2, "advanc": 2, "algorithm": 2, "ensembl": [2, 3], "improv": 2, "addition": 2, "incorpor": 2, "review": 2, "histor": 2, "enhanc": 2, "power": 2, "demonstr": 2, "driven": 2, "approach": 2, "economi": 2, "sklearn": 3, "standardscal": 3, "onehotencod": 3, "compos": 3, "columntransform": 3, "model_select": 3, "train_test_split": 3, "gridsearchcv": 3, "linear_model": 3, "mean_squared_error": 3, "mean_absolute_error": 3, "copi": 3, "importlib": 3, "reload": 3, "create_summari": 3, "calculate_metr": 3, "display_result": 3, "One": 3, "hot": 3, "encod": 3, "categor": 3, "cat_featur": 3, "enc": 3, "spars": 3, "drop": 3, "encoded_featur": 3, "fit_transform": 3, "encoded_features_df": 3, "get_feature_names_out": 3, "append": 3, "axi": 3, "concat": 3, "defin": 3, "continu": 3, "your": 3, "srv": 3, "amef": 3, "lib": 3, "python3": 3, "site": 3, "_encod": 3, "868": 3, "wa": [3, 4], "renam": 3, "sparse_output": 3, "ignor": [3, 4], "unless": 3, "leav": 3, "its": 3, "80": 3, "20": 3, "x_train": 3, "x_test": 3, "y_train": 3, "y_test": 3, "test_siz": 3, "random_st": 3, "1234": 3, "x_valid": 3, "y_valid": 3, "331": 3, "shape": 3, "25248": 3, "6312": 3, "7891": 3, "two": [3, 4], "three": 3, "pipe": 3, "streamlin": 3, "fit": 3, "pipe_nam": 3, "item": 3, "paramet": 3, "cross": 3, "cv_param_grid_al": 3, "rf__min_samples_leaf": 3, "lasso__alpha": 3, "logspac": 3, "knn_imputer__n_neighbor": 3, "ridge__alpha": 3, "For": 3, "store": [3, 4], "score": 3, "valid_err": 3, "tuned_pipelin": 3, "ypred_valid": 3, "cv_param_grid": 3, "startswith": 3, "tupl": 3, "named_step": 3, "pipe_search": 3, "deepcopi": 3, "highest": 3, "best_model": 3, "y_pred_test": 3, "test_ms": 3, "test_ma": 3, "test_r2": 3, "tabl": 3, "format": 3, "estim": 3, "get": [3, 4], "best_rf_model": 3, "best_estimator_": 3, "feature_importances_": 3, "feature_nam": 3, "sort": 3, "descend": 3, "order": 3, "match": 3, "correspond": 3, "sorted_indic": 3, "argsort": 3, "sorted_import": 3, "sorted_feature_nam": 3, "barh": 3, "align": 3, "gca": 3, "invert_yaxi": 3, "top": 3, "statist": 4, "259": 4, "spring": 4, "2023": 4, "due": 4, "wednesdai": 4, "05": 4, "00pm": 4, "pt": 4, "prof": 4, "p\u00e9rez": 4, "gsi": 4, "sapienza": 4, "depart": 4, "uc": 4, "berkelei": 4, "assign": 4, "worth": 4, "maximum": 4, "point": 4, "group": 4, "free": 4, "form": 4, "develop": 4, "complet": 4, "onlin": 4, "ask": 4, "some": 4, "interest": 4, "question": 4, "document": 4, "manner": 4, "ve": 4, "been": 4, "dure": 4, "cours": 4, "depend": 4, "size": 4, "mai": 4, "choos": 4, "judgment": 4, "few": 4, "dozen": 4, "megabyt": 4, "problem": 4, "someth": 4, "gigabyt": 4, "too": 4, "big": 4, "put": 4, "If": 4, "isn": 4, "remot": 4, "cach": 4, "adequ": 4, "subsequ": 4, "don": 4, "consid": 4, "implic": 4, "long": 4, "term": 4, "sourc": 4, "one": 4, "Will": 4, "year": 4, "also": 4, "wai": 4, "zenodo": 4, "standalon": 4, "write": 4, "proper": 4, "those": 4, "docstr": 4, "input": 4, "shown": 4, "exampl": 4, "detail": 4, "high": 4, "qualiti": 4, "pure": 4, "itself": 4, "choic": 4, "cleanest": 4, "fluid": 4, "workflow": 4, "whether": 4, "script": 4, "sure": 4, "least": 4, "note": 4, "must": 4, "properli": 4, "feel": 4, "dictat": 4, "break": 4, "down": 4, "mani": 4, "reason": 4, "conveni": 4, "read": 4, "There": 4, "hard": 4, "fast": 4, "rule": 4, "just": 4, "paragraph": 4, "scientif": 4, "paper": 4, "readabl": 4, "time": 4, "length": 4, "etc": 4, "criteria": 4, "essai": 4, "stephen": 4, "wolfram": 4, "creator": 4, "mathematica": 4, "part": 4, "inspir": 4, "thought": 4, "what": 4, "written": 4, "comput": 4, "chain": 4, "analys": 4, "intermedi": 4, "being": 4, "thei": 4, "need": 4, "recomput": 4, "scratch": 4, "mind": 4, "reus": 4, "report": 4, "disk": 4, "discuss": 4, "look": 4, "back": 4, "instruct": 4, "refresh": 4, "summar": 4, "It": 4, "referenc": 4, "think": 4, "assumpt": 4, "about": 4, "justifi": 4, "control": 4, "over": 4, "acquisit": 4, "measur": 4, "like": 4, "fall": 4, "broad": 4, "purview": 4, "exploratori": 4, "propos": 4, "hypothesi": 4, "justif": 4, "acquir": 4, "expect": 4, "spend": 4, "lot": 4, "try": 4, "focu": 4, "author": 4, "end": 4, "brief": 4, "team": 4, "member": 4, "did": 4, "stori": 4, "sentenc": 4, "per": 4, "suffic": 4, "agre": 4, "languag": 4, "standard": 4, "journal": 4, "while": 4, "principl": 4, "same": 4, "reserv": 4, "anyon": 4, "effort": 4, "previou": 4, "homework": 4, "necessari": 4, "recommend": 4, "new": 4, "virtual": 4, "updat": 4, "librari": 4, "forget": 4, "seed": 4, "anyth": 4, "stochast": 4, "compon": 4, "collabor": 4, "ad": 4, "commit": 4, "messag": 4, "branch": 4, "readm": 4, "md": 4, "short": 4, "self": 4, "descript": 4, "motiv": 4, "conduct": 4, "relev": 4, "instal": 4, "autom": 4, "A": 4, "start": 4, "licens": 4, "explicitli": 4, "state": 4, "condit": 4, "appli": 4, "see": 4, "github": 4, "strongli": 4, "suggest": 4, "victoria": 4, "stodden": 4, "enabl": 4, "research": 4, "innov": 4, "she": 4, "text": 4, "media": 4, "materi": 4, "maxim": 4, "reshar": 4, "credit": 4, "guarante": 4, "gitignor": 4, "prevent": 4, "automat": 4, "inclus": 4, "unwant": 4, "avoid": 4, "noisi": 4, "statu": 4, "output": 4, "thing": 4, "know": 4, "won": 4, "actual": 4, "whole": 4, "element": 4, "plai": 4, "central": 4, "role": 4, "class": 4, "add": 4, "semest": 4, "possibl": 4, "idea": 4, "hw07": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"project": [0, 4], "group": 0, "28": 0, "airbnb": 0, "europ": 0, "dataset": 0, "analysi": [0, 2, 4], "websit": 0, "repositori": [0, 4], "structur": [0, 4], "setup": 0, "instal": 0, "usag": 0, "launch": 0, "interact": 0, "binder": 0, "licens": 0, "data": [1, 2, 3, 4], "preprocess": [1, 2], "A": 2, "comprehens": 2, "us": 2, "machin": 2, "learn": 2, "techniqu": 2, "introduct": 2, "descript": 2, "exploratori": 2, "featur": 2, "engin": 2, "model": [2, 3], "build": [2, 3], "evalu": [2, 3], "result": 2, "interpret": 2, "conclus": 2, "test": [3, 4], "final": 4, "origin": 4, "deliver": 4, "function": 4, "your": 4, "code": 4, "notebook": 4, "support": 4, "main": 4, "narr": 4, "reproduc": 4, "good": 4, "practic": 4, "grade": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})