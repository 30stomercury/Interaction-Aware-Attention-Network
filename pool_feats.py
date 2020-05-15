from data import mean_pool
import joblib


# argument
parser = ArgumentParser()
parser.add_argument('--input_file', default='data/feats.pkl', type=str)
parser.add_argument('--output_file', default='data/feats_pooled.pkl', type=str)
parser.add_argument('--step', default=2, type=int)
parser.add_argument('--feat_dim', default='39', type=int)
parser.add_argument('--max_step', default=2500, type=int)
args = parser.parse_args()

feats = joblib.load(args.input_file)
feat_pooled = mean_pool(feats, step=args.step, dim=args.feat_dim, max_step=args.max_step)

joblib.dump(feat_pooled, args.output_file)
