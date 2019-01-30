import itertools
import numpy as np


def simpleSelect(db):
    queries = ""
    for table, value in db.items():
        queries = queries+"SELECT * FROM "+table+";\n"
    return queries

def simpleCount(db):
    queries = ""
    for table, value in db.items():
        queries = queries+"SELECT COUNT(*) FROM "+table+";\n"
    return queries

def simpleDistinct(db):
    queries = ""
    for table, value in db.items():
        for column in value:
            queries = queries+"SELECT DISTINCT("+column+") FROM "+table+";\n"
    return queries

def oneWayJoin(con):
    queries = ""
    for i in con:
        for join_pair in itertools.combinations(i, 2):
            queries = queries + "SELECT * FROM " + join_pair[0].split(".")[0] + ", "+join_pair[1].split(".")[0]+" WHERE "+join_pair[0]+"="+join_pair[1]+";\n"
    return queries

def towWayJoin(db, con):
        queries = ""
        tables = ""
        for t, value in db.items():
            if tables is "":
                tables = t
            else:
                tables += ","+t
        for i in con:
            for join_pair in itertools.combinations(i, 2):
                for j in con:
                    if i is not j:
                        for join_pair2 in itertools.combinations(j, 2):
                            if not ((join_pair[0].split(".")[0]==join_pair2[0].split(".")[0] and join_pair[1].split(".")[0]==join_pair2[1].split(".")[0]) or (join_pair[0].split(".")[0]==join_pair2[1].split(".")[0] and join_pair[1].split(".")[0]==join_pair2[0].split(".")[0])):
                                queries = queries + "SELECT * FROM " + tables + " WHERE "+join_pair[0]+"="+join_pair[1]+" AND "+join_pair2[0]+"="+join_pair2[1]+";\n"
        return queries

def rangeQueries(ranges,db,con,max_val):
    queries = ""
    queries_text = simpleSelect(db)+simpleCount(db)+oneWayJoin(con)+towWayJoin(db,con)
    queries_array= queries_text.split("\n")[:-1]
    for val,rows in max_val.items():
        real_ranges = np.multiply(ranges,int(int(val)/10))
        for row in rows:
            for r in real_ranges:
                where = " WHERE "+row+">="+str(r[0])+" AND "+row+"<"+str(r[1])
                andd = " AND "+row+">="+str(r[0])+" AND "+row+"<"+str(r[1])
                for query in queries_array:
                    table_ind=query.replace("SELECT * FROM ","").replace("SELECT COUNT(*) FROM ","").replace(", ","").replace(",","").replace(";",";;")
                    print(table_ind)
                    print(table_ind[0])
                    if table_ind[0]==row[0] or table_ind[1]==row[0] or table_ind[2]==row[0]:
                        if query.__contains__("WHERE"):
                            queries+=query.replace(";",andd+";\n")
                        else:
                            queries+=query.replace(";", where + ";\n")
    return queries


text=simpleSelect(db)+simpleCount(db)+simpleDistinct(db)
text+=oneWayJoin(connections)+towWayJoin(db,connections)

ranges=[[0,1],[3,4],[6,7],[8,9],[2,4],[5,7],[2,7],[5,10],[1,8]]
text+=rangeQueries(ranges,db,connections,max_val)

print(text)
output = open("queries/queries.txt","w")
output.write(text)
output.close()
