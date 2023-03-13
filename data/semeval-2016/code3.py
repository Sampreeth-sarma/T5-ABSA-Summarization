logs = ["supply item1:2 100",
        "supply item2:3 60",
        "sell item1:1",
        "sell item1:4",
        "sell item2:2",
        "return item2:1 60 defect",
        "sell item2:2"
        ]

log_dict = {}
sell_logs = []
for each in logs:
    log_splt = each.split()

    if log_splt[0] == "supply":
        item_name = log_splt[1].split(":")[0]
        item_count = int(log_splt[1].split(":")[1])
        item_price = int(log_splt[2])
        if item_name not in log_dict:
            log_dict[item_name] = [[item_count], [item_price]]
        else:
            if item_price in log_dict[item_name][1]:
                log_dict[item_name][0][log_dict[item_name][1].index(item_price)] += item_count
            else:
                log_dict[item_name][0].append(item_count)
                log_dict[item_name][1].append(item_price)
                item_count_list, item_price_list = [list(v) for v in zip(*sorted(zip(log_dict[item_name][1], log_dict[item_name][0])))]
                log_dict[item_name][0] = item_count_list
                log_dict[item_name][1] = item_price_list
    elif log_splt[0] == "sell":
        req_item_name = log_splt[1].split(":")[0]
        req_item_count = int(log_splt[1].split(":")[1])
        if req_item_name not in log_dict:
            sell_logs.append(-1)
        else:
            if sum(log_dict[req_item_name][0]) >= req_item_count:
                for each_item_count in log_dict[req_item_name][0]:
                    if each_item_count >= req_item_count:
                        remaining = each_item_count - req_item_count
                        sell_logs.append(req_item_count * log_dict[req_item_name][1][0])
                        if remaining == 0 and len(log_dict[req_item_name][0]) == 1:
                            log_dict.pop(req_item_name)
                        elif remaining == 0:
                            log_dict[req_item_name][0] = log_dict[req_item_name][0][1:]
                            log_dict[req_item_name][1] = log_dict[req_item_name][1][1:]
                        else:
                            log_dict[req_item_name][0][0] -= req_item_count
            else:
                sell_logs.append(-1)
    elif log_splt[0] == "return":
        ret_item_name = log_splt[1].split(":")[0]
        ret_item_count = int(log_splt[1].split(":")[1])
        ret_item_price = int(log_splt[2])
        reason = log_splt[3]

        if reason != "defect":
            if ret_item_name not in log_dict:
                log_dict[ret_item_name] = [[ret_item_count], [ret_item_price]]
            else:
                if ceil(float(ret_item_price) * float(0.8)) in log_dict[ret_item_name][1]:
                    log_dict[ret_item_name][0][log_dict[ret_item_name][1].index(ceil(float(ret_item_price) * float(0.8)))] += ret_item_count
                else:
                    log_dict[ret_item_name][0].append(ret_item_count)
                    log_dict[ret_item_name][1].append(ceil(float(ret_item_price) * float(0.8)))
                    ret_item_count_list, ret_item_price_list = [list(v) for v in zip(*sorted(
                        zip(log_dict[ret_item_name][1], log_dict[ret_item_name][0])))]
                    log_dict[ret_item_name][0] = ret_item_count_list
                    log_dict[ret_item_name][1] = ret_item_price_list

print(sell_logs)

