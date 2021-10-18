# -*- coding: utf-8 -*-

# 一、Apriori&FPgrowth measures函数封包
# 加载必要的包，设置DataFrame参数
import pandas as pd
import math, typing, numbers, itertools, datetime
from efficient_apriori import rules
from efficient_apriori.itemsets import apriori_gen
from efficient_apriori.itemsets import itemsets_from_transactions, ItemsetCount
from pyfpgrowth.pyfpgrowth import FPTree
import sys

heading_properties = [('font-size', '12px')]
cell_properties = [('font-size', '10px')]
center_properties = [('text-align', 'center')]
dfstyle = [dict(selector="th", props=heading_properties),
           dict(selector="td", props=cell_properties),
           dict(selector="th", props=center_properties)]


# Apriori算法    
class Rule_ex(rules.Rule):
    """
    Calculate phi_correlation.
    """
    
    @property
    def correlation(self):
        """
        Phi_correlation.
        """
        try:
            phi_numerator = self.num_transactions * self.count_full - self.count_lhs * self.count_rhs
            phi_denominator = math.sqrt(self.count_lhs * self.count_rhs 
                                        * (self.num_transactions - self.count_lhs)
                                        * (self.num_transactions - self.count_rhs))
            return phi_numerator / phi_denominator
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None
        
    def __str__(self):
        """
        Printing of a rule.
        """
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))


def _ap_genrules(
    itemset: tuple,
    H_m: typing.List[tuple],
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_conf: float,
    num_transactions: int,
):
    def count(itemset):
        return itemsets[len(itemset)][itemset]

    # If H_1 is so large that calling `apriori_gen` will produce right-hand
    # sides as large as `itemset`, there will be no right hand side
    # This cannot happen, so abort if it will
    if len(itemset) <= (len(H_m[0]) + 1):
        return

    # Generate left-hand itemsets of length k + 1 if H is of length k
    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    # For every possible right hand side
    for h_m in H_m:
        # Compute the right hand side of the rule
        lhs = tuple(sorted(set(itemset).difference(set(h_m))))

        # If the confidence is high enough, yield the rule, else remove from
        # the upcoming recursive generator call
        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule_ex(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)

    # Unless the list of right-hand sides is empty, recurse the generator call
    if H_m_copy:
        yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, num_transactions)

def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_confidence: float,
    num_transactions: int,
    verbosity: int = 0,
):
    # Validate user inputs
    if not ((0 <= min_confidence <= 1) and isinstance(min_confidence, numbers.Number)):
        raise ValueError("`min_confidence` must be a number between 0 and 1.")

    if not ((num_transactions >= 0) and isinstance(num_transactions, numbers.Number)):
        raise ValueError("`num_transactions` must be a number greater than 0.")

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")

    # For every itemset of a perscribed size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        # For every itemset of this size
        for itemset in itemsets[size].keys():

            # Special case to capture rules such as {others} -> {1 item}
            for removed in itertools.combinations(itemset, 1):

                # Compute the left hand side
                remaining = set(itemset).difference(set(removed))
                lhs = tuple(sorted(remaining))

                # If the confidence is high enough, yield the rule
                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule_ex(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )
            # Generate combinations to start off of. These 1-combinations will
            # be merged to 2-combinations in the function `_ap_genrules`
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(itemset, H_1, itemsets, min_confidence, num_transactions)

    if verbosity > 0:
        print("Rule generation terminated.\n")

def apriori(
    transactions: typing.Union[typing.List[tuple], typing.Callable],
    min_support: float = 0.5,
    min_confidence: float = 0.5,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    itemsets, num_trans = itemsets_from_transactions(
        transactions,
        min_support,
        max_length,
        verbosity,
        output_transaction_ids,
    )
    if itemsets and isinstance(next(iter(itemsets[1].values())), ItemsetCount):
        itemsets_for_rules = _convert_to_counts(itemsets)
    else:
        itemsets_for_rules = itemsets

    rules = generate_rules_apriori(itemsets_for_rules, min_confidence, num_trans, verbosity)
    return itemsets, list(rules)

def _convert_to_counts(itemsets):
    itemsets_counts = {}
    for size, sets in itemsets.items():
        itemsets_counts[size] = {i: c.itemset_count for i, c in sets.items()}
    return itemsets_counts

# FP-growth算法
def find_frequent_patterns(transactions, support_threshold):
    tree = FPTree(transactions, support_threshold, None, None)
    tr_patterns = tree.mine_patterns(support_threshold)
    tr_source = tree.frequent
    tr_source_keys = dict(zip(list(tr_source.keys()), 
                              [(tr_key,) for tr_key in list(tr_source.keys())]))
    tr_source_target = dict((tr_source_keys[key], value) for (key, value) in tr_source.items())
    return {**tr_patterns, **tr_source_target}

def generate_association_rules(patterns, confidence_threshold, transactions):
    rules = []
    num_transactions = len(transactions)
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support
           
                    if confidence >= confidence_threshold:
                        support = upper_support / num_transactions
                        try:
                            lower_support_r = patterns[consequent]
                            lift = (num_transactions * upper_support) / (lower_support * lower_support_r)
                            phi_numerator = num_transactions * upper_support - lower_support * lower_support_r
                            phi_denominator = math.sqrt(lower_support * lower_support_r
                                                        * (num_transactions - lower_support)
                                                        * (num_transactions - lower_support_r))
                            phi_corr = phi_numerator / phi_denominator
                            rule_value = (consequent, support, confidence, lift, phi_corr)
                            rule_key_value = (antecedent, rule_value)
                            rules.append(rule_key_value)
                        except:
                            sys.exc_clear()
    return rules

# 算法执行
def data_loader(df, groupby_key, groupby_column, algorithm="apr"):
    """
    Load data.
    Parameters:
        algorithm: apr -- Apriori
                   fp -- FP-growth
        groupby_key: pandas df.groupby (key)
        groupby_column: pandas df.groupby [column]
    """
    datadict = dict(list(df.groupby(groupby_key)[groupby_column]))
    datakeys = datadict.keys()
    transactions = []
    for key in datakeys:
        if algorithm == "apr":
            transactions.append(tuple(set([x for x in datadict[key]])))
        elif algorithm == "fp":
            transactions.append(list(set([x for x in datadict[key]])))
        else:
            raise KeyError
    return transactions

def apriori_main(transactions, min_support, min_confidence, printorder=0):
    """
    执行Apriori算法，
    If printorder=0：按Lift值从大到下进行排序，输出前20个最强规则
    If printorder=1：按φ的绝对值从大到下进行排序，输出前20个最强规则
    """
    starttime = datetime.datetime.now()
    apr_patterns, apr_rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    endtime = datetime.datetime.now()
    print("Apriori算法本次用时为 {}s".format((endtime - starttime)))
    
    if (printorder==0):
        apr_rules = sorted(apr_rules, key=lambda rule: rule.lift,reverse=True)
    elif (printorder==1):
        apr_rules = sorted(apr_rules, key=lambda rule: abs(rule.correlation),reverse=True)

    # apr_rules = apr_rules[:20]
    col_1 = []
    support = []
    confidence = []
    lift = []
    correlation = []
    for index in range(len(apr_rules)):
        col_1.append(apr_rules[index])
        support.append(apr_rules[index].support)
        confidence.append(apr_rules[index].confidence)
        lift.append(apr_rules[index].lift)
        correlation.append(apr_rules[index].correlation)
    apr_dict = {"Rule":col_1, "Support":support, "Confidence":confidence, "Lift":lift, "Correlation":correlation}
    apr_df = pd.DataFrame(apr_dict, columns=["Rule", "Support", "Confidence", "Lift", "Correlation"])
    return apr_df

def fp_growth_main(transactions, min_support, min_confidence, printorder=0):
    """
    执行FP-growth算法，
    If printorder=0：按Lift值从大到下进行排序，输出前20个最强规则
    If printorder=1：按φ的绝对值从大到下进行排序，输出前20个最强规则
    """
    starttime = datetime.datetime.now()
    fp_patterns = find_frequent_patterns(transactions, min_support*len(transactions))
    fp_rules = generate_association_rules(fp_patterns, min_confidence, transactions)
    endtime = datetime.datetime.now()
    print("FP-growth算法本次用时为 {}s".format((endtime - starttime)))
    
    if (printorder==0):
        fp_rules = sorted(fp_rules, key=lambda rule: abs(rule[1][3]),reverse=True)
    elif (printorder==1):
        fp_rules = sorted(fp_rules, key=lambda rule: abs(rule[1][4]),reverse=True)
    
    # fp_rules = fp_rules[:20]
    col_1 = []
    support = []
    confidence = []
    lift = []
    correlation = []
    for index in range(len(fp_rules)):
        col_1.append("{} -> {}".format(set(fp_rules[index][0]), set(fp_rules[index][1][0])))
        support.append(fp_rules[index][1][1])
        confidence.append(fp_rules[index][1][2])
        lift.append(fp_rules[index][1][3])
        correlation.append(fp_rules[index][1][4])
    fp_dict = {"Rule":col_1, "Support":support, "Confidence":confidence, "Lift":lift, "Correlation":correlation}
    fp_df = pd.DataFrame(fp_dict, columns=["Rule", "Support", "Confidence", "Lift", "Correlation"])
    return fp_df

# 二、数据集 1
# 2.0 数据概览
# 数据集路径
filepath_1 = ".\Power Mart.xlsx"
Power_Mart = pd.read_excel(filepath_1)
Power_Mart.head()

# 以Order ID为单位，计算总事务数
len(Power_Mart.loc[:,'Order ID'].unique()) 

# 以Product Name为单位，计算不同的商品数
len(Power_Mart.loc[:,'Product Name'].unique())

"""
Product Name的unique个数太多，没必要对其在进行预处理： 一方面是个数太多NLP存在难度，
另一方面是后续有Category和Sub-Category的任务。
"""

# 2.1 参数调整、关联规则输出、算法时间代价比较
# 2.1.1 第一组参数：minsup=0.4 minconf=0.4
# 参数
min_support = 0.4 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 2.1.2 第二组参数：minsup=0.01 minconf=0.4
# 参数
min_support = 0.01 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 2.1.3 第三组参数：minsup=0.0005 minconf=0.4
# 参数
min_support = 0.0005 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 2.1.4 第四组参数：minsup=0.0002 minconf=0.4
# 参数
min_support = 0.0002 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df[0:20].style.set_table_styles(dfstyle)
Power_Mart_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df[0:20].style.set_table_styles(dfstyle)

Power_Mart_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 2.1.5 第五组参数：minsup=0.0002 minconf=0.3
# 参数
min_support = 0.0002 # 支持度阈值
min_confidence = 0.3 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df[0:20].style.set_table_styles(dfstyle)
Power_Mart_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df[0:20].style.set_table_styles(dfstyle)

Power_Mart_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 2.2 规则图示
# 见R语言代码

# 2.3 最强规则及兴趣度指标计算
# 采用的是2.1.5的第五组参数
# 参数
min_support = 0.0002 # 支持度阈值
min_confidence = 0.3 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Product Name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df[0:20].style.set_table_styles(dfstyle)

# 2.4 商品类上不同子集关联分析
# 2.4.0 数据输入和变量指定
# 数据集路径
path = ".\Power Mart.xlsx"
PM = pd.read_excel(path)
# 变量指定
groupby_key = "Order ID" # 汇总索引
groupby_column = "Sub-Category" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# 2.4.1 根据 Region 变量分成多个子集合
# 支持度和置信度的阈值参数
min_support = 0.005
min_confidence = 0.3
regions = list(set(PM['Region']))
region_cut = [PM[PM['Region'] == region] for region in regions]

# Apriori算法
PM_R_apr_transactions = [data_loader(region, groupby_key, groupby_column, algorithm="apr") for region in region_cut]
PM_R_apr = [apriori_main(PM_R_apr_trans, min_support, min_confidence) for PM_R_apr_trans in PM_R_apr_transactions]

# FP-growth算法
PM_R_fp_transactions = [data_loader(region, groupby_key, groupby_column, algorithm="fp") for region in region_cut]
PM_R_fp = [fp_growth_main(PM_R_fp_trans, min_support, min_confidence) for PM_R_fp_trans in PM_R_fp_transactions]

writer_R = pd.ExcelWriter('Region.xlsx')
for i in range(4):
    PM_R_fp[i].to_excel(writer_R, sheet_name=regions[i], encoding='UTF-8')
writer_R.save()

# 2.4.2 根据Segment变量分成多个子集合
segments = list(set(PM['Segment']))
segment_cut = [PM[PM['Segment'] == segment] for segment in segments]

# Apriori算法
PM_S_apr_transactions = [data_loader(segment, groupby_key, groupby_column, algorithm="apr") for segment in segment_cut]
PM_S_apr = [apriori_main(PM_S_apr_trans, min_support, min_confidence) for PM_S_apr_trans in PM_S_apr_transactions]

# FP-growth算法
PM_S_fp_transactions = [data_loader(segment, groupby_key, groupby_column, algorithm="fp") for segment in segment_cut]
PM_S_fp = [fp_growth_main(PM_S_fp_trans, min_support, min_confidence) for PM_S_fp_trans in PM_S_fp_transactions]

writer_S = pd.ExcelWriter('Segment.xlsx')
for i in range(3):
    PM_S_fp[i].to_excel(writer_S, sheet_name=segments[i], encoding='utf-8')
writer_S.save()

# 2.4.3 根据 Ship Mode 变量分成多个子集合
shipmodes = list(set(PM['Ship Mode']))
shipmode_cut = [PM[PM['Ship Mode'] == shipmode] for shipmode in shipmodes]
# 支持度和置信度的阈值参数
min_support = 0.008
min_confidence = 0.3

# Apriori算法
PM_M_apr_transactions = [data_loader(shipmode, groupby_key, groupby_column, algorithm="apr") for shipmode in shipmode_cut]
PM_M_apr = [apriori_main(PM_M_apr_trans, min_support, min_confidence) for PM_M_apr_trans in PM_M_apr_transactions]

# FP-growth算法
PM_M_fp_transactions = [data_loader(shipmode, groupby_key, groupby_column, algorithm="fp") for shipmode in shipmode_cut]
PM_M_fp = [fp_growth_main(PM_M_fp_trans, min_support, min_confidence) for PM_M_fp_trans in PM_M_fp_transactions]

writer_M = pd.ExcelWriter('Ship Mode.xlsx')
for i in range(4):
    PM_M_fp[i].to_excel(writer_M, sheet_name=shipmodes[i], encoding='utf-8')
writer_M.save()

# 2.5 商品类上总体关联分析
filepath_1 = ".\Power Mart.xlsx"
Power_Mart = pd.read_excel(filepath_1)
Power_Mart.head()

# 以Order ID为单位，计算总事务数
len(Power_Mart.loc[:,'Order ID'].unique()) 

# 以Product Name为单位，计算不同的商品数
len(Power_Mart.loc[:,'Sub-Category'].unique())

# 2.5.1 参数调整、关联规则输出、算法时间代价比较
# 参数
min_support = 0.001 # 支持度阈值
min_confidence = 0.5 # 置信度阈值
groupby_key = "Order ID" # 汇总索引
groupby_column = "Sub-Category" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Power_Mart_apr_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="apr")
Power_Mart_apr_df = apriori_main(Power_Mart_apr_transactions, min_support, min_confidence)
Power_Mart_apr_df[0:20].style.set_table_styles(dfstyle)

# FP-growth算法
Power_Mart_fp_transactions = data_loader(Power_Mart, groupby_key, groupby_column, algorithm="fp")
Power_Mart_fp_df = fp_growth_main(Power_Mart_fp_transactions, min_support, min_confidence)
Power_Mart_fp_df[0:20].style.set_table_styles(dfstyle)

# 2.5.2 规则图示
# 见R语言代码

# 三、数据集 2
# 3.0 数据概览与合并
# 数据集路径
filepath_2 = ".\\order_products_train.csv"
Order_products_train = pd.read_csv(filepath_2)
Order_products_train.head()

# 原数据中观测个数
len(Order_products_train.iloc[:,0])

# 计算事务数
len(Order_products_train.loc[:,'order_id'].unique()) 

# 计算项集大小
len(Order_products_train.loc[:,'product_id'].unique()) 

filepath_3 = ".\\products.xlsx"
products = pd.read_excel(filepath_3)
products.head()

Order_products_train = pd.merge(Order_products_train, products, how = 'inner', on = 'product_id')
Order_products_train.head()

# 3.1 参数调整、关联规则输出、算法时间代价比较
# 3.1.1 第一组参数：minsup=0.4 minconf=0.4
min_support = 0.4 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Order_products_train_apr_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="apr")
Order_products_train_apr_df = apriori_main(Order_products_train_apr_transactions, min_support, min_confidence)
Order_products_train_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 3.1.2 第二组参数：minsup=0.05 minconf=0.4
min_support = 0.05 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Order_products_train_apr_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="apr")
Order_products_train_apr_df = apriori_main(Order_products_train_apr_transactions, min_support, min_confidence)
Order_products_train_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 3.1.3 第三组参数：minsup=0.01 minconf=0.4
min_support = 0.01 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Order_products_train_apr_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="apr")
Order_products_train_apr_df = apriori_main(Order_products_train_apr_transactions, min_support, min_confidence)
Order_products_train_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 3.1.4 第四组参数：minsup=0.003 minconf=0.4
min_support = 0.003 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Order_products_train_apr_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="apr")
Order_products_train_apr_df = apriori_main(Order_products_train_apr_transactions, min_support, min_confidence)
Order_products_train_apr_df.style.set_table_styles(dfstyle)
Order_products_train_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df.style.set_table_styles(dfstyle)
Order_products_train_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# 3.1.5 第五组参数：minsup=0.003 minconf=0.3
min_support = 0.003 # 支持度阈值
min_confidence = 0.3 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth

# Apriori算法
Order_products_train_apr_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="apr")
Order_products_train_apr_df = apriori_main(Order_products_train_apr_transactions, min_support, min_confidence)
Order_products_train_apr_df[0:20].style.set_table_styles(dfstyle)
Order_products_train_apr_df.shape[0] # 统计满足支持度和置信度要求的规则条数

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df[0:20].style.set_table_styles(dfstyle)
Order_products_train_fp_df.shape[0] # 统计满足支持度和置信度要求的规则条数
# 5综上，选择第五组参数进行后续处理！

# 3.2 规则图示
# 见R语言代码

# 3.3 最强规则及兴趣度指标计算
# 采用3.1.5的第五组参数
min_support = 0.003 # 支持度阈值
min_confidence = 0.3 # 置信度阈值
groupby_key = "order_id"# 汇总索引
groupby_column = "product_name" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth

# FP-growth算法
Order_products_train_fp_transactions = data_loader(Order_products_train, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df[0:20].style.set_table_styles(dfstyle)
Order_products_train_fp_df.shape[0] #统计满足支持度和置信度要求的规则条数

# 3.4 将商品概化为商品类（根据department字段） ：第（4）问
# 画图
import datetime
from functools import reduce
from pathlib import Path

from bqplot import *
from bqplot.marks import Graph
from efficient_apriori import apriori
from ipywidgets import (
    AppLayout,
    Button,
    FloatLogSlider,
    FloatSlider,
    HBox,
    IntSlider,
    Label,
    Layout,
    SelectMultiple,
    Textarea,
    TwoByTwoLayout,
    VBox,
)


class Arulesviz:
    def __init__(
        self,
        transactions,
        min_sup,
        min_conf,
        min_lift,
        max_sup=1.0,
        min_slift=0.1,
        products_to_drop=[],
    ):
        self.rules = []
        self.transactions = transactions
        self.min_lift = min_lift
        self.min_slift = min_slift
        self.min_slift = min_slift or min_lift
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.max_sup = max_sup
        self.products_to_in = []
        self.products_to_out = products_to_drop
        self._hovered_product = None

    def _standardized_lift(self, rule, s=None, c=None):
        """
        Parameters
        ----------
        rule:
              Target rule
        s: float
           Support treshold user for rule mining
        c: float
           Confidence treshold user for rule mining
        """
        s = s or self.min_sup
        c = c or self.min_conf
        prob_A = getattr(rule, "support") / getattr(rule, "confidence")
        prob_B = getattr(rule, "confidence") / getattr(rule, "lift")
        mult_A_and_B = prob_A * prob_B
        L = max(
            1 / prob_A + 1 / prob_B - 1 / (mult_A_and_B),
            s / mult_A_and_B,
            c / prob_B,
            0,
        )
        U = min(1 / prob_A, 1 / prob_B)
        slift = (getattr(rule, "lift") - L) / (U - L)
        return slift

    def create_rules(self, drop_products=True, max_sup=None):
        max_sup = max_sup or self.max_sup
        tr = self.transactions
        if drop_products:
            to_drop = set(self.products_to_out)
            tr = [set(x) - to_drop for x in tr]
            tr = [x for x in tr if x]
        _, self.rules = apriori(
            tr, min_support=self.min_sup, min_confidence=self.min_conf
        )
        for rule in self.rules:
            setattr(rule, "slift", self._standardized_lift(rule))
        self.rules = self.filter_numeric(
            "support", max_sup, self.rules, should_be_lower=True
        )
        self._max_sup = max([x.support for x in self.rules])
        self._max_conf = max([x.confidence for x in self.rules])

    def filter_numeric(self, atr, val, rules, should_be_lower=False):
        rules = rules
        if should_be_lower:
            return [x for x in rules if getattr(x, atr) < val]
        return [x for x in rules if getattr(x, atr) > val]

    def filter_drop_if_name_in(self, vals, rules, lhs=True, rhs=True):
        rules = rules
        vals = set(vals)
        f = lambda x: not any(
            [(lhs and (vals & set(x.lhs))), (rhs and (vals & set(x.rhs)))]
        )
        return list(filter(f, rules))

    def filter_drop_if_name_out(self, vals, rules, lhs=True, rhs=True):
        rules = rules
        vals = set(vals)
        f = lambda x: any(
            [(lhs and (vals & set(x.lhs))), (rhs and (vals & set(x.rhs)))]
        )
        return list(filter(f, rules))

    def get_unique_products(self, rules):
        rules = rules
        return reduce(
            lambda x, y: (x if isinstance(x, set) else set(x.lhs) | set(x.rhs))
            | set(y.lhs)
            | set(y.rhs),
            rules,
        )

    def create_graph(self, rules):
        rules = rules
        nodes = []
        links = []
        colors = []
        name_to_id = {}
        already_seen = set()
        for sr in rules:
            current_comb = tuple(sorted(set(sr.lhs) | set(sr.rhs)))
            if current_comb in already_seen:
                continue
            else:
                already_seen.add(current_comb)
            # node_size = max(min(sr.lift * 10, 30), 5)
            nodes.append(
                {
                    "label": f".",
                    "shape": "circle",
                    "shape_attrs": {"r": max(min(sr.lift, 7), 2)},
                    "is_rule": True,
                    "tooltip": str(sr),
                }
            )
            colors.append("black")
            rule_id = len(nodes) - 1

            for node_name in sr.lhs:
                l_node_id = name_to_id.get(node_name, None)
                if l_node_id == None:
                    nodes.append(
                        {
                            "label": node_name,
                            "shape": "rect",
                            "is_rule": False,
                            "shape_attrs": {
                                "width": 6 * len(node_name) + 8,
                                "height": 20,
                            },
                        }
                    )
                    colors.append("white")
                    l_node_id = len(nodes) - 1
                    name_to_id[node_name] = l_node_id
                links.append({"source": l_node_id, "target": rule_id, "value": sr.lift})

            for node_name in sr.rhs:
                r_node_id = name_to_id.get(node_name, None)
                if r_node_id == None:
                    nodes.append(
                        {
                            "label": node_name,
                            "shape": "rect",
                            "is_rule": False,
                            "shape_attrs": {
                                "width": 6 * len(node_name) + 8,
                                "height": 20,
                            },
                        }
                    )
                    r_node_id = len(nodes) - 1
                    name_to_id[node_name] = r_node_id
                    colors.append("white")
                links.append({"source": rule_id, "target": r_node_id, "value": sr.lift})
        return nodes, links, colors

    def replot_graph(self):
        sub_rules = self.filter_numeric("lift", self.min_lift, rules=self.rules)
        sub_rules = self.filter_numeric("support", self.min_sup, rules=sub_rules)
        sub_rules = self.filter_numeric("slift", self.min_slift, rules=sub_rules)
        sub_rules = self.filter_numeric("confidence", self.min_conf, rules=sub_rules)
        sub_rules = self.filter_drop_if_name_in(self.products_to_out, rules=sub_rules)
        sub_rules = self.filter_drop_if_name_out(self.products_to_in, rules=sub_rules)
        (
            self.graph.node_data,
            self.graph.link_data,
            _,  # self.graph.colors,
        ) = self.create_graph(sub_rules)

    def handler_products_out_filter(self, value):
        self.products_to_out = value["new"]
        self.replot_graph()

    def setup_products_out_selector(self):
        self.selector_products_out = SelectMultiple(
            options=sorted(self.get_unique_products(self.rules)),
            value=[],
            rows=10,
            # description="Drop",
            disabled=False,
        )
        self.selector_products_out.observe(self.handler_products_out_filter, "value")

    def handler_products_in_filter(self, value):
        self.products_to_in = value["new"]
        self.replot_graph()

    def setup_products_in_selector(self):
        self.selector_products_in = SelectMultiple(
            options=sorted(self.get_unique_products(self.rules)),
            value=sorted(self.get_unique_products(self.rules)),
            rows=10,
            # description="Include",
            disabled=False,
        )
        self.products_to_in = sorted(self.get_unique_products(self.rules))
        self.selector_products_in.observe(self.handler_products_in_filter, "value")

    def set_slider_value(self, value):
        setattr(self, getattr(value["owner"], "description"), value["new"])
        self.replot_graph()

    def setup_lift_slider(self):
        name = "lift"
        setattr(
            self,
            f"slider_{name}",
            FloatLogSlider(
                value=getattr(self, f"min_{name}"),
                min=-0.5,
                max=1.5,
                step=0.05,
                base=10,
                description=f"min_{name}",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".3f",
            ),
        )
        getattr(self, f"slider_{name}").observe(self.set_slider_value, "value")

    def setup_conf_slider(self):
        name = "conf"
        setattr(
            self,
            f"slider_{name}",
            FloatSlider(
                value=getattr(self, f"min_{name}"),
                min=0.0,
                max=self._max_conf,
                step=0.0001,
                base=10,
                description=f"min_{name}",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".5f",
            ),
        )
        getattr(self, f"slider_{name}").observe(self.set_slider_value, "value")

    def setup_slift_slider(self):
        name = "slift"
        setattr(
            self,
            f"slider_{name}",
            FloatSlider(
                value=getattr(self, f"min_{name}"),
                min=0.0,
                max=1.0,
                step=0.0001,
                base=10,
                description=f"min_{name}",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".5f",
            ),
        )
        getattr(self, f"slider_{name}").observe(self.set_slider_value, "value")

    def setup_sup_slider(self):
        name = "sup"
        setattr(
            self,
            f"slider_{name}",
            FloatSlider(
                value=getattr(self, f"min_{name}"),
                min=0.0,
                max=self._max_sup,
                step=0.0001,
                base=10,
                description=f"min_{name}",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".5f",
            ),
        )
        getattr(self, f"slider_{name}").observe(self.set_slider_value, "value")

    def _save_graph_img(self, b):
        self.fig.save_png(
            f"arulesviz_{datetime.datetime.now().isoformat().replace(':','-').split('.')[0]}.png"
        )

    def setup_graph_to_img_button(self):
        self.graph_to_img_button = Button(description="Save img!")
        self.graph_to_img_button.on_click(self._save_graph_img)

    def plot_graph(
        self,
        width=1000,
        height=750,
        charge=-200,
        link_type="arc",
        directed=True,
        link_distance=100,
    ):
        fig_layout = Layout(width=f"{width}px", height=f"{height}px")
        nodes, links, colors = self.create_graph(
            self.filter_numeric("lift", self.min_lift, rules=self.rules)
        )
        # xs = LinearScale(min=0, max=1000)
        # ys = LinearScale(min=0, max=750)
        cs = ColorScale(scheme="Reds")
        self.graph = Graph(
            node_data=nodes,
            link_data=links,
            # colors=colors,
            charge=charge,
            link_type=link_type,
            directed=directed,
            link_distance=link_distance,
            # scales={'color': cs}
        )
        margin = dict(top=-60, bottom=-60, left=-60, right=-60)
        self.fig = Figure(
            marks=[self.graph],
            layout=Layout(width=f"{width}px", height=f"{height}px"),
            fig_margin=dict(top=0, bottom=0, left=0, right=0),
            legend_text={"font-size": 7},
        )

        # tooltip = Tooltip(fields=["foo"], formats=["", "", ""])
        # self.graph.tooltip = tooltip

        # self.graph.on_hover(self.hover_handler)
        self.graph.on_element_click(self.hover_handler)
        self.graph.on_background_click(self.clean_tooltip)
        self.graph.interactions = {"click": "tooltip"}
        self.setup_sup_slider()
        self.setup_lift_slider()
        self.setup_conf_slider()
        self.setup_slift_slider()
        self.setup_products_in_selector()
        self.setup_products_out_selector()
        self.setup_graph_to_img_button()
        self.setup_product_tooltip()
        return VBox(
            [
                HBox(
                    [
                        self.selector_products_in,
                        self.selector_products_out,
                        VBox(
                            [
                                getattr(self, "slider_lift"),
                                getattr(self, "slider_slift"),
                                getattr(self, "slider_conf"),
                                getattr(self, "slider_sup"),
                            ]
                        ),
                        getattr(self, "graph_to_img_button"),
                    ]
                ),
                self.fig,
            ]
        )

    def clean_tooltip(self, x, y):
        self.graph.tooltip = None

    def plot_scatter(
        self,
        products=[],
        min_width=600,
        min_height=600,
        max_width=600,
        max_height=600,
        with_toolbar=True,
        display_names=False,
    ):
        if products:
            sub_rules = self.filter_drop_if_name_out(products, self.rules)
        else:
            sub_rules = self.rules
        data_x = [np.round(x.support * 100, 3) for x in sub_rules]
        data_y = [np.round(x.confidence * 100, 3) for x in sub_rules]
        color = [np.round(x.lift, 4) for x in sub_rules]
        names = [str(sr) for sr in sub_rules]
        sc_x = LinearScale()
        sc_y = LinearScale()
        sc_color = ColorScale(scheme="Reds")
        ax_c = ColorAxis(
            scale=sc_color,
            tick_format="",
            label="Lift",
            orientation="vertical",
            side="right",
        )
        tt = Tooltip(fields=["name"], formats=[""])
        scatt = Scatter(
            x=data_x,
            y=data_y,
            color=color,
            scales={"x": sc_x, "y": sc_y, "color": sc_color},
            tooltip=tt,
            names=names,
            display_names=display_names,
        )
        ax_x = Axis(scale=sc_x, label="Sup*100")
        ax_y = Axis(scale=sc_y, label="Conf*100", orientation="vertical")
        m_chart = dict(top=50, bottom=70, left=50, right=100)
        fig = Figure(
            marks=[scatt],
            axes=[ax_x, ax_y, ax_c],
            fig_margin=m_chart,
            layout=Layout(
                min_width=f"{min_width}px",
                min_height=f"{min_height}px",
                max_width=f"{max_width}px",
                max_height=f"{max_height}px",
            ),
        )
        if with_toolbar:
            toolbar = Toolbar(figure=fig)
            return VBox([fig, toolbar])
        else:
            return fig

    def setup_product_tooltip(self, products=[]):
        self.graph.tooltip = self.plot_scatter(products)
        if len(products) == 1:
            self.graph.tooltip.title = products[-1]
        else:
            self.graph.tooltip.title = "Products scatter"

    def hover_handler(self, qq, content):
        product = content.get("data", {}).get("label", -1)
        is_rule = content.get("data", {}).get("tooltip", None)
        if product != self._hovered_product:
            if is_rule:
                self._hovered_product = content.get("data", {}).get("tooltip", None)
                self.graph.tooltip = Textarea(
                    content.get("data", {}).get("tooltip", None)
                )
                self.graph.tooltip_location = "center"
            else:
                self._hovered_product = product
                self.setup_product_tooltip([product])
                self.graph.tooltip_location = "center"

# 1、读取数据并将数据集按照key进行join，对数据集的department和aisles两个label进行基础的描述性统计分析
order = pd.read_csv("./order_products_train.csv")
department = pd.read_csv("./departments.csv")
aisles = pd.read_csv("./aisles.csv")
product = pd.read_csv("./products.csv",encoding='ISO-8859-1')
## 以order为主表，其他表left join连接
order_data = pd.merge((pd.merge((pd.merge(order, product, on = 'product_id', how ='left')),department,on='department_id',how='left')),aisles,on='aisle_id',how='left')
len(order_data[order_data.department == 'deli'].loc[:,'order_id'].unique())
print(len(order_data.loc[:,'order_id'].unique()))
print(len(order_data.loc[:,'department'].unique()))
print(len(order_data.loc[:,'aisle'].unique()))
print(order_data.loc[:,'department'].unique())
print(order_data.loc[:,'aisle'].unique())

"""
在department这个类别中，客户购买比例最多的是produce，产品包括各种fresh vegetables,fresh fruits,packaged vegetables fruits；客户购买比例第二多的是dairy eggs,产品包括各种乳制品、鸡蛋、黄油、豆制品等。
"""

# 2、department
basket_Sub = list(order_data.groupby('order_id')['department'])
dataset=[]
for i in range(len(basket_Sub)):
    dataset.append(list(basket_Sub[i][1]))
g = Arulesviz(dataset, 0.1, 0.4, 0, products_to_drop=[])
g.create_rules()
g.plot_scatter()
# 可以看出lift大的点大多集中在support=0.1，confidence=[0.3,0.6]左右，因此将参数设定为min_support=0.1，min_confidence=0.4

# 参数
min_support = 0.1 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id" # 汇总索引
groupby_column = "department" # 需要汇总的列
# 注：data_loader需要传入 algorithm 参数, "apr":Apriori, "fp":FP-growth
# FP-growth算法
Order_products_train_fp_transactions = data_loader(order_data, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df[0:20].style.set_table_styles(dfstyle)

"""
超市的经营中，陈列是一项重要的技术，超市将产品A和B放在一起，不仅要考虑A对B的促销作用，还要考虑B对A的促销作用。在department维度上，选择参数min_support=0.1，min_confidence=0.4筛选出来的20条规则（按照lift进行排序），虽然这19条规则在confidence上表现的不错，但是lift都在1左右，说明在LHS的条件下对RHS事件发生率的提升度都在1左右，可以说两个条件没有任何关联，可以说并不是很有趣的关联规则。

但可以相对比较看看，
"""
g = Arulesviz(dataset, 0.1, 0.4, 1.5, products_to_drop = [])
g.create_rules()
g.plot_graph(width = 1000, directed = True, charge = -1000, link_distance = 20)

# 3.5 将商品概化为商品类（根据aisle字段）
basket_Sub = list(order_data.groupby('order_id')['aisle'])
dataset=[]
for i in range(len(basket_Sub)):
    dataset.append(list(basket_Sub[i][1]))
g = Arulesviz(dataset, 0.2, 0.4, 0, products_to_drop = [])
g.create_rules()
g.plot_scatter()
# 可以看出lift大的点大多集中在support=0.2，confidence=[0.4,0.8]左右，因此将参数设定为min_support=0.2，min_confidence=0.4

# 参数
min_support = 0.2 # 支持度阈值
min_confidence = 0.4 # 置信度阈值
groupby_key = "order_id" # 汇总索引
groupby_column = "aisle" # 需要汇总的列
# 注：data_loader需要传入algorithm参数, "apr":Apriori, "fp":FP-growth
# FP-growth算法
Order_products_train_fp_transactions = data_loader(order_data, groupby_key, groupby_column, algorithm="fp")
Order_products_train_fp_df = fp_growth_main(Order_products_train_fp_transactions, min_support, min_confidence)
Order_products_train_fp_df[0:20].style.set_table_styles(dfstyle)

"""
在department维度上，选择参数min_support=0.2，min_confidence=0.4筛选出来的11条规则（按照lift进行排序），
集中在fresh fruits，fresh vegetables，packaged vegetables fruits三类产品之间，其中lift最高的两条规则显示，
买了packaged vegetables fruits会购买fresh fruits和fresh vegetables，购买了fresh fruits和fresh vegetables
也会购买packaged vegetables fruits，起到了相互促销的作用，这符合常理。

"""

g = Arulesviz(dataset, 0.2, 0.4, 0, products_to_drop=[])
g.create_rules()
g.plot_graph(width = 1000, directed = True, charge = -1000, link_distance = 20)