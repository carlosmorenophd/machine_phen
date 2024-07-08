from obregon_1516_1617_svr import do_run_svr
from obregon_1516_1617_rf import do_run_rf
from obregon_1516_1617_xgboost import do_run_xgb


if __name__ == "__main__":
    do_run_svr(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_svr.csv")
    # do_run_rf(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_rf.csv")
    # do_run_xgb(do_basic=True, do_heatmap=True, do_pca=True, file_name="result_test_xgb.csv")
