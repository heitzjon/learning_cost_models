import psycopg2
import multiprocessing
import time

# EXECUTE QUERY
def exec_query(query):
    try:
        conn = psycopg2.connect(host="", database="", user="", password="")
    except:
        print("I am unable to connect to the database")
    #print(query)
    cursor = conn.cursor()
    cursor.execute("""EXPLAIN ANALYZE """ + query)
    rows = cursor.fetchall()

    #print(rows)

    executionTime = rows[len(rows) - 1][0].split(' ')[2]
    planningTime = rows[len(rows) - 2][0].split(' ')[2]
    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedCost = row0[0].split('..')[1]
    estimatedRows = row0[1].replace("rows=", "")
    estimatedWidth = row0[2].replace("width=", "").replace(")", "")
    actualRows = row0[5].replace("rows=", "")
    actualLoops = row0[6].replace("loops=", "").replace(")", "")
    output_line = query.replace("\n","") + "|" + estimatedCost + "|" + estimatedRows + "|" + estimatedWidth + "|" + actualRows + "|" + actualLoops + "|" + executionTime + "|" + planningTime + "\n"
    print(query)
    print(estimatedCost + "|" + estimatedRows + "|" + estimatedWidth + "|" + actualRows + "|" + actualLoops + "|" + executionTime + "|" + planningTime + "\n")
    outputfile = open("", "a")
    outputfile.write(output_line)
    outputfile.close()


#START AT QUERY
query_nr=-1

#READ FILE
input = open("queries/queries.txt","r")
queries = input.readlines()
output = open("","w")




#DB CONNECTION
#try:
#    conn = psycopg2.connect(host="localhost",database="imdbload", user="postgres", password="admin")
#except:
#    print("I am unable to connect to the database")

#cur = conn.cursor()

if query_nr <= 0: output.write("query|est_cost|est_row|est_width|act_row|act_loops|exec_time|planning_time\n")
output.close()
i=0
for query in queries:
    if i <= query_nr:
        i += 1
    else:
        print(i)
        p = multiprocessing.Process(target=exec_query, name="EXECUTEQUERY", args=(query,))
        p.start()
        p.join(900)

        # If thread is active
        if p.is_alive():
            print(str(i)+ "  is running... let's kill it!")

            # Terminate foo
            p.terminate()
            p.join()
        i += 1

output.close()
input.close()

