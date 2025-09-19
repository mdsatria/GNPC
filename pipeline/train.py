import path_project

from gnpc.default_modules import *
from gnpc.gnn.data.dataset import make_graph_dloader
from gnpc.gnn.network.loss import NLLDeepSurv
from gnpc.gnn.network.model import GraphSurv
from gnpc.gnn.train.loop import TrainGNNSurvival
from gnpc.gnn.train.utils import select_optimiser
from gnpc.utils.io import *
from gnpc.utils.logger import LogMsg
from gnpc.utils.misc import get_feature_len, get_graph_type, set_seed


def write_file(fpath, message):
    with open(fpath, "a") as file:
        file.writelines(f"{message}\n")


def main(config, seed=21):
    torch.cuda.empty_cache()
    set_seed(seed)

    endpoint = str(config["train_config"]["endpoint"])
    graph_type = get_graph_type(
        deep_feature=config["graph_config"]["embedding"],
        nuclei_feature=config["graph_config"]["is_morph"],
        is_weighted=config["graph_config"]["is_weighted"],
    )
    num_feature = get_feature_len(config)

    dir_graph_train = os.path.join(
        config["graph_config"]["dir_graph_train"], graph_type
    )
    dir_graph_val = os.path.join(config["graph_config"]["dir_graph_val"], graph_type)
    dir_graph_test = os.path.join(config["graph_config"]["dir_graph_test"], graph_type)

    dir_root_save = os.path.join(
        str(config["train_config"]["dir_save"]),
        endpoint.upper(),
        str(config["train_config"]["run_name"]),
        str(config["train_config"]["exp_id"]),
    )

    dir_models = os.path.join(dir_root_save, "models")
    dir_patstart = os.path.join(dir_root_save, "patient_strat")
    if not os.path.exists(dir_models):
        os.makedirs(dir_models)
    if not os.path.exists(dir_patstart):
        os.makedirs(dir_patstart)

    fpath_config = os.path.join(
        dir_root_save,
        "config.json",
    )
    fpath_result = os.path.join(
        dir_root_save,
        "results.json",
    )

    save_json(config, fpath_config)

    fpath_log = os.path.join(dir_root_save, "training.log")
    logger = LogMsg(fpath_log=fpath_log)

    df_train = pd.read_csv(config["graph_config"]["csv_train"])
    df_val = pd.read_csv(config["graph_config"]["csv_val"])
    df_test = pd.read_csv(config["graph_config"]["csv_test"])
    logger.msg(f"Train data: {df_train.shape[0]}")
    logger.msg(f"Validation data: {df_val.shape[0]}  Test data: {df_test.shape[0]}")
    logger.msg("Calculating train graph.X mean and variance")

    dloader_train, global_mean, global_std = make_graph_dloader(
        df=df_train,
        dir_graph=dir_graph_train,
        is_train=True,
        global_mean=None,
        global_std=None,
        batch_size=config["train_config"]["batch_size"],
        num_worker=config["train_config"]["num_worker"],
    )
    dloader_val = make_graph_dloader(
        df=df_val,
        dir_graph=dir_graph_val,
        is_train=False,
        global_mean=global_mean,
        global_std=global_std,
        batch_size=config["train_config"]["batch_size"],
        num_worker=config["train_config"]["num_worker"],
    )
    dloader_test = make_graph_dloader(
        df=df_test,
        dir_graph=dir_graph_test,
        is_train=False,
        global_mean=global_mean,
        global_std=global_std,
        batch_size=config["train_config"]["batch_size"],
        num_worker=config["train_config"]["num_worker"],
    )
    model = GraphSurv(
        num_feature=num_feature,
        is_weighted=config["graph_config"]["is_weighted"],
        is_multimodal=config["graph_config"]["is_multimodal"],
        conv=config["model_config"]["m_conv"],
        activation_conv=config["model_config"]["m_act_conv"],
        activation_fcn=config["model_config"]["m_act_fcn"],
        use_last_activation_fcn=config["model_config"]["m_use_last"],
        use_norm_fcn=config["model_config"]["m_use_norm_fcn"],
        return_attention=False,
    )
    criterion = NLLDeepSurv(
        l2=config["train_config"]["l2_nll"], device=config["train_config"]["device"]
    )
    optimiser = select_optimiser(
        model=model,
        optim_name=config["train_config"]["optimiser"],
        lr=config["train_config"]["lr"],
        weight_decay=config["train_config"]["weight_decay"],
    )
    if config["train_config"]["scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=config["train_config"]["epochs"] // 4,
            eta_min=5e-4,
            verbose=False,
        )
    else:
        scheduler = None

    trainer = TrainGNNSurvival(
        model=model,
        optimiser=optimiser,
        criterion=criterion,
        device=config["train_config"]["device"],
        scheduler=scheduler,
    )

    best_val_cindex = 0
    dct_metrics = {}
    logger.msg(f"{endpoint.upper()} Train-test start")

    for epoch in range(config["train_config"]["epochs"]):
        train_loss, ps_train, cut_off = trainer.train(
            current_epoch=epoch + 1, dloader=dloader_train
        )
        loss_val, ps_val, _ = trainer.infer(
            dloader=dloader_val,
            is_train=False,
            cut_off_in=cut_off,
            use_best_model=False,
        )
        if ps_val.c_index > best_val_cindex:
            best_val_cindex = ps_val.c_index
            dct_metrics = {
                "train_cindex": ps_train.c_index,
                "val_cindex": ps_val.c_index,
                "train_pvalue": ps_train.p_value,
                "val_pvalue": ps_val.p_value,
                "cut_off": cut_off,
            }
            trainer.set_best_model()
            torch.save(trainer.best_model, os.path.join(dir_models, f"model.pt"))

        log_msg = f"{endpoint.upper()} {config['graph_config']['embedding'].upper()} {config['train_config']['exp_id']}"
        log_msg = f"{log_msg} Epoch:{epoch+1}/{config['train_config']['epochs']} "
        logger.msg(log_msg)
        log_msg = f"    TRAIN    CI:{ps_train.c_index:.3f}"
        log_msg = f"{log_msg} loss:{train_loss:.3f}"
        log_msg = f"{log_msg} pvalue:{ps_train.p_value:.3f}"
        logger.msg(log_msg)
        log_msg = f"    VAL    CI:{ps_val.c_index:.3f}"
        log_msg = f"{log_msg} loss:{loss_val:.3f}"
        log_msg = f"{log_msg} pvalue:{ps_val.p_value:.3f}"
        logger.msg(log_msg)

    logger.msg(f"Training finished\n")
    loss_test, ps_test, _ = trainer.infer(
        dloader=dloader_test,
        is_train=False,
        cut_off_in=dct_metrics["cut_off"],
        use_best_model=True,
    )
    dct_metrics["test_cindex"] = ps_test.c_index
    dct_metrics["test_pvalue"] = ps_test.p_value
    logger.msg(f"Test finished")

    for k, v in dct_metrics.items():
        logger.msg(f"{k}: {v}")
    save_json(dct_metrics, fpath_result)
    logger.msg(f"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_json", type=str)
    args = parser.parse_args()

    params = load_json(args.config_json)
    main(config=params)
