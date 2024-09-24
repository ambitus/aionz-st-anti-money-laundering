def get_colnames(params):
    colnames = []
    
    # add raw features names
    colnames.append("transactionID")
    colnames.append("sourceAccountID")
    colnames.append("destinationAccountID")
    colnames.append("timestamp")
    colnames.append("amount sent")
    colnames.append("currency sent")
    colnames.append("amount received")
    colnames.append("currency received")
    colnames.append("payment format")
    
    # add features names for the graph patterns
    for pattern in ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle']:
        if pattern in params:
            if params[pattern]:
                bins = len(params[pattern +'_bins'])
                if pattern in ['fan', 'degree']:
                    for i in range(bins-1):
                        colnames.append(pattern+"-in length "+str(params[pattern +'_bins'][i]))
                    colnames.append(pattern+" length >="+str(params[pattern +'_bins'][i+1]))
                    for i in range(bins-1):
                        colnames.append(pattern+"-out length "+str(params[pattern +'_bins'][i]))
                    colnames.append(pattern+"-out length >="+str(params[pattern +'_bins'][i+1]))
                else:
                    for i in range(bins-1):
                        colnames.append(pattern+" length "+str(params[pattern +'_bins'][i]))
                    colnames.append(pattern+" length >="+str(params[pattern +'_bins'][i+1]))

    vert_feat_names = ["fan","deg","ratio","avg","sum","min","max","median","var","skew","kurtosis"]

    coldict = {
      0: "transID",
      1: "srcAccID",
      2: "destAccID",
      3: "timestamp",
      4: "ammountSent",
      5: "currencySent",
      6: "ammountReceived",
      7: "currencyReceived",
      7: "format"
    }
    
    # add features names for the vertex statistics
    for orig in ['source', 'dest']:
        for direction in ['out', 'in']:
            # add fan, deg, and ratio features
            for k in [0, 1, 2]:
                if k in params["vertex_stats_feats"]:
                    feat_name = orig + "_" + vert_feat_names[k] + "_" + direction
                    colnames.append(feat_name)
            for col in params["vertex_stats_cols"]:
                colname = "col" + str(col)
                if col in coldict:
                    colname = coldict[col]
                # add avg, sum, min, max, median, var, skew, and kurtosis features
                for k in [3, 4, 5, 6, 7, 8, 9, 10]:
                    if k in params["vertex_stats_feats"]:
                        feat_name = orig + "_" + vert_feat_names[k] + "_" + colname + "_" + direction
                        colnames.append(feat_name)

    return colnames