import path_project
from tqdm.contrib.concurrent import thread_map

from gnpc.default_modules import *
from gnpc.gnn.data.dataset import make_graph_dloader
from gnpc.gnn.network.model import GraphSurv
from gnpc.utils.io import *
from gnpc.utils.logger import LogMsg
from gnpc.utils.misc import get_feature_len, get_graph_type_from_json


def generate_attention_score_cohort(
    fpath_model, fpath_config, dir_save, is_internal=False
):
    config = load_json(fpath_config)

    df_train = pd.read_csv(config["graph_config"]["csv_train"])
    dir_graph_train = config["graph_config"]["dir_graph_internal"]
    if is_internal:
        df_test = pd.read_csv(config["graph_config"]["csv_internal"])
        dir_graph_test = config["graph_config"]["dir_graph_internal"]
    else:
        df_test = pd.read_csv(config["graph_config"]["csv_external"])
        dir_graph_test = config["graph_config"]["dir_graph_external"]
    feature_len = get_feature_len(config)
    print(feature_len)
    graph_type = get_graph_type_from_json(config)
    model = GraphSurv(
        num_feature=feature_len,
        is_weighted=config["graph_config"]["is_weighted"],
        is_multimodal=config["graph_config"]["is_multimodal"],
        conv=config["model_config"]["m_conv"],
        activation_conv=config["model_config"]["m_act_conv"],
        activation_fcn=config["model_config"]["m_act_fcn"],
        use_last_activation_fcn=config["model_config"]["m_use_last"],
        use_norm_fcn=config["model_config"]["m_use_norm_fcn"],
        return_attention=True,
    )
    model.load_state_dict(torch.load(fpath_model))

    dir_graph_train = os.path.join(dir_graph_train, graph_type)
    dir_graph_test = os.path.join(dir_graph_test, graph_type)

    _, global_mean, global_std = make_graph_dloader(
        dir_graph=dir_graph_train, df=df_train, batch_size=1, is_train=True
    )
    dloader_test = make_graph_dloader(
        dir_graph=dir_graph_test,
        df=df_test,
        batch_size=1,
        is_train=False,
        global_mean=global_mean,
        global_std=global_std,
    )
    print("infer process")
    model.eval()
    model.cuda()
    with torch.no_grad():
        for data in tqdm(dloader_test):
            graph = data["graph"]
            clinical_data = data["clinical_data"]
            output, dct_attention = model(graph.cuda(), clinical_data.cuda())
            for key, tensorlist in dct_attention.items():
                for ix, tensor in enumerate(tensorlist):
                    dct_attention[key][ix] = tensor.cpu().numpy()
            dct_attention["edge_index"] = graph["edge_index"].cpu().numpy()
            fpath_save = os.path.join(dir_save, f"{data['patient'][0]}.pkl")
            save_pickle(dct_attention, fpath_save)
    return len(dloader_test)


class SlideLevelAttention:
    def __init__(
        self,
        csv_dataset,
        dir_attention,
        base_graph,
    ):
        self.csv_dataset = csv_dataset
        self.dir_attention = dir_attention
        self.base_graph = base_graph

        self.get_patient_dct()

    def get_patient_dct(self, col_patient="patient", col_slide="wsi"):
        df = pd.read_csv(self.csv_dataset)
        dct_patient = {}
        for patient in df[col_patient].unique().tolist():
            slide_list = df[df[col_patient] == patient][col_slide].values.tolist()
            dct_patient[str(patient)] = [pathlib.Path(x).stem for x in slide_list]

        self.dct_patient_slide = dct_patient

    def load_patient(self, patient):
        self.patient = patient
        self.fpath_attention = os.path.join(
            self.dir_attention,
            f"{self.patient}.pkl",
        )
        self.dir_graph = os.path.join(self.base_graph, self.patient)
        self.fpath_graph_patient = os.path.join(self.dir_graph, f"{self.patient}.pt")
        self.slide_list = self.dct_patient_slide[self.patient]
        self.fpath_graph_slides = [
            os.path.join(self.dir_graph, "slide", f"{x}.pt") for x in self.slide_list
        ]
        self.graph_patient = torch.load(self.fpath_graph_patient)
        self.graph_slide = {
            pathlib.Path(x).stem: torch.load(x) for x in self.fpath_graph_slides
        }
        self.dct_attention = load_pickle(self.fpath_attention)

    def extract_attention_slide(self):
        if len(self.slide_list) > 1:
            dct_attention_slide = {x: {} for x in self.slide_list}
            for layer, attention_node in self.dct_attention.items():
                if "layer" not in layer:
                    continue
                else:
                    patient_node_indices = attention_node[0]
                    patient_node_attention_score = attention_node[1]
                    prev_max_node_index = 0
                    for slide_name, graph_slide in self.graph_slide.items():
                        len_node = graph_slide.x.shape[0]
                        dct_attention_slide[slide_name]["node_len"] = len_node
                        node_slide = np.array(range(len_node)) + prev_max_node_index
                        mask = np.isin(patient_node_indices, node_slide)
                        node_indices = patient_node_indices[mask] - prev_max_node_index
                        node_attention_score = patient_node_attention_score[mask]
                        dct_attention_slide[slide_name][layer] = [
                            node_indices,
                            node_attention_score,
                        ]
                        prev_max_node_index += len_node

            return dct_attention_slide
        else:
            return {self.slide_list[0]: self.dct_attention}


def check_make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def distangle_process(patient_id, csv_dataset, dir_attention, base_graph):
    gres = SlideLevelAttention(
        csv_dataset=csv_dataset,
        dir_attention=dir_attention,
        base_graph=base_graph,
    )
    gres.load_patient(patient_id)
    dct_attention_slide = gres.extract_attention_slide()
    for slide_name, attention_score in dct_attention_slide.items():
        fpath = os.path.join(dir_save, f"{slide_name}.pkl")
        save_pickle(attention_score, fpath)
    return len(gres.slide_list)


if __name__ == "__main__":

    dir_final_exp = "/data/codes/github/npc_graph_stme/data/final_results_mc"
    base_dir_save = "/data/codes/github/npc_graph_stme/data/attention"
    base_dir_graph = "/mnt/storage/project_data/graphsurv/graphs/"
    base_clinical_file = "/mnt/storage/project_data/graphsurv/csv"

    logger = LogMsg(os.path.join(base_dir_save, "generate_attention_score.log"))

    """Generate attention per graph(patient)"""
    logger.msg("Generate attention per graph(patient)")
    for endpoint in ["dmfs", "lrfs", "os"]:
        fpath_config = os.path.join(
            dir_final_exp,
            endpoint,
            f"config.json",
        )
        graph_type = get_graph_type_from_json(load_json(fpath_config))
        fpath_model = os.path.join(dir_final_exp, endpoint, "model.pt")
        for cohort in ["cuhk", "sysucc"]:
            if cohort == "cuhk":
                is_internal = False
            else:
                is_internal = True
            dir_save = os.path.join(
                base_dir_save,
                "patient_level",
                endpoint,
                cohort,
            )
            check_make_dir(dir_save)
            len_patient = generate_attention_score_cohort(
                fpath_model, fpath_config, dir_save, is_internal=is_internal
            )
            logger.msg(f"{endpoint} {cohort} total patient: {len_patient}")

    """ Distangle patient level attention score to slide level"""
    logger.msg("\nDistangle patient level attention score to slide level")
    for endpoint in ["dmfs", "lrfs", "os"]:
        for cohort in ["cuhk", "sysucc"]:
            dir_graph = os.path.join(base_dir_graph, cohort, graph_type)
            dir_attention_patient_level = os.path.join(
                base_dir_save, "patient_level", endpoint, cohort
            )
            dir_attention_slide_level = os.path.join(
                base_dir_save, "slide_level", endpoint, cohort
            )
            csv_dataset = os.path.join(base_clinical_file, cohort, "final_clean.csv")
            patient_list = [
                pathlib.Path(x).stem for x in os.listdir(dir_attention_patient_level)
            ]

            check_make_dir(dir_attention_slide_level)
            total_processed_slide = 0
            gres = SlideLevelAttention(
                csv_dataset=csv_dataset,
                dir_attention=dir_attention_patient_level,
                base_graph=dir_graph,
            )
            for patient in tqdm(patient_list):
                gres.load_patient(patient)
                dct_attention_slide = gres.extract_attention_slide()
                for slide_name, attention_score in dct_attention_slide.items():
                    fpath = os.path.join(dir_attention_slide_level, f"{slide_name}.pkl")
                    save_pickle(attention_score, fpath)
                    total_processed_slide += 1
            logger.msg(
                f"{endpoint} {cohort} total patient: {len(patient_list)} total slide: {total_processed_slide}"
            )
