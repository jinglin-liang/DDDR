import os
import argparse
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import DataManager, setup_seed, count_parameters
from methods import get_learner


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # General settings
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
    parser.add_argument('--save_dir', type=str, default="outputs", help='save data')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--g_sigma', type=float, default=0, help='sigma of updata g dp')
    parser.add_argument('--classifer_dp', type=float, default=0, help='dp add to classifer')
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="ours", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0.5, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')

    # Target settings
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data')
    
    # DDDR settings
    parser.add_argument('--w_kd', type=float, default=10., help='for kd loss')
    parser.add_argument('--w_ce_pre', type=float, default=0.5, help='for syn ce loss')
    parser.add_argument('--w_scl', type=float, default=1., help='use supervised contrastive learning loss')
    parser.add_argument('--com_round_gen', type=int, default=10, help='communication rounds')
    parser.add_argument('--g_local_train_steps', type=int, default=50, help='local train steps')
    parser.add_argument('--config', type=str, default="ldm/ldm_dddr.yaml", help='config of diffusion')
    parser.add_argument('--ldm_ckpt', type=str, default="models/ldm/text2img-large/model.ckpt", help='checkpoint path of latent diffusion model')
    parser.add_argument('--no_scale_lr', action='store_true', help='scale_lr')
    parser.add_argument('--g_local_bs', type=int, default=12, help='local batch size')
    parser.add_argument('--n_iter', type=int, default=5, help='generate syndata iter')
    parser.add_argument('--syn_image_path', type=str, default=None, help='resume syndata path')
    parser.add_argument('--pre_size', type=int, default=200, help='pre syndata size for per class')
    parser.add_argument('--cur_size', type=int, default=50, help='cur syndata size for per class')
    parser.add_argument('--save_cls_embeds', action='store_true', help='scale_lr')
    
    args = parser.parse_args()
    return args


def train(args):
    setup_seed(args["seed"])
    data_manager = DataManager(
        args["dataset"],
        True,
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    learner = get_learner(args["method"], args)
    for _ in range(data_manager.nb_tasks):
        print("All params: {}, Trainable params: {}".format(
            count_parameters(learner._network), 
            count_parameters(learner._network, True)))
        learner.incremental_train(data_manager) # train for one task
        learner.eval_task()
        learner.after_task()
        learner.log_metrics()


if __name__ == '__main__':
    args = args_parser()
    args.num_class = 200 if args.dataset=="tiny_imagenet" else 100 
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls
    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.exp_name = f"beta_{args.beta}_tasks_{args.tasks}_seed_{args.seed}_sigma_{args.g_sigma}_{args.exp_name}"
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, args.exp_name)
    args = vars(args)
    train(args)
