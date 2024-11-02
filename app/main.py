# from obregon_1516_1617_svr import do_run_svr
# from obregon_1516_1617_rf import do_run_rf
# from obregon_1516_1617_xgboost import do_run_xgb
# from do_run.obregon_1516_1617 import launch_all
# from notebooks.no_ndvi.process_no_ndvi_wheater import do_run
# import math

# def combinaciones(n, k):
#   return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

from notebooks.best_ndvi.process_best_ndvi import run

if __name__ == "__main__":
    run()
    
    # C(2054, 1) + C(2054, 2) + C(2054, 3) + ... + C(2054, 2054)
#     C(8, 1) = 8
# C(8, 2) = 28
# C(8, 3) = 56
# C(8, 4) = 70
# C(8, 5) = 56
# C(8, 6) = 28
# C(8, 7) = 8
# C(8, 8) = 1
# 255

# 1025! = 9.33262154439441e+157.
# estrellas 4 x 10^11 

    # resultado = combinaciones(2050, 2)
    # print(resultado)


    # launch_all()
    # print("Finish")
    # # do_run_svr(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_svr.csv")
    # do_run_rf(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_rf.csv")
    # do_run_xgb(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_xgb.csv")

