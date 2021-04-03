import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.stats import spearmanr, kendalltau, beta


def load_players(fpath):
    players = pd.read_pickle(fpath)
    players = pd.DataFrame(list(players.values()))
    players['name'] = players['name'] + ' ' + players['surname']
    players = players[['id', 'name']]
    return players


def load_tournaments(fpath):
    tournaments = pd.read_pickle(fpath)
    tournaments = pd.DataFrame(list(tournaments.values()))
    tournaments['year'] = pd.to_datetime(tournaments['dateStart']).apply(pd.to_datetime, utc=True).dt.year
    tournaments['type'] = tournaments['type'].apply(lambda t: t['name'])
    tournaments = tournaments[['id', 'name', 'year', 'type']].reset_index(drop=True)
    return tournaments


def load_results(fpath, tournaments):
    results = pd.read_pickle(fpath)
    team_results = []

    for tournament_id, teams in tqdm(results.items()):
        if tournament_id not in tournaments['id'].values:
            continue
        if (len(teams) < 2):
            continue

        tournament_results = []
        for team in teams:
            if ('mask' not in team) or \
               (team['mask'] is None) or \
               ('X' in team['mask']) or \
               ('?' in team['mask']) or \
               (len(team['teamMembers']) == 0):
                continue

            maks = np.array(list(map(int, team['mask'].replace('X', '0').replace('?', '0'))))

            team_result = {}
            team_result['game_id'] = tournament_id
            team_result['team_id'] = team['team']['id']
            team_result['questions'] = maks
            team_result['wins'] = sum(maks)
            team_result['total'] = len(maks)
            team_result['position'] = team['position']
            team_result['members'] = [m['player']['id'] for m in team['teamMembers']]
            team_result['num_players'] = len(team_result['members'])

            tournament_results.append(team_result)
            
        if len(set([r['total'] for r in tournament_results])) == 1:
            team_results.extend(tournament_results)

    team_results = pd.DataFrame(team_results)
    team_results['year'] = team_results.merge(tournaments, left_on='game_id', right_on='id')['year']

    return team_results


def correlation_score(y_game):
    y_true = y_game['position'].values
    y_pred = y_game['team_rating'].values
    return pd.Series({'spearman': -spearmanr(y_true, y_pred)[0], 
                      'kendall': -kendalltau(y_true, y_pred)[0]})
    
    
def mean_correlation_score(y_true, y_pred):
    y_games = y_true.merge(y_pred, on=['game_id', 'team_id'])
    y_stats = y_games.groupby('game_id').apply(correlation_score)
    score = y_stats.values.mean(axis=0)
    return {'spearman': round(score[0], 3), 'kendall': round(score[1], 3)}


def top_score(top100_true, top100_pred):
    players_true = set(top100_true['player_id'].values)
    players_pred = set(top100_pred['player_id'].values)
    players_common = players_pred.intersection(players_true)
    players_all = players_pred.union(players_true)
    
    players_join = top100_true.merge(top100_pred, on='player_id')
    y_true = players_join['player_rating_x'].values
    y_pred = players_join['player_rating_y'].values
    iou = len(players_common)/len(players_all)
    
    iou_score = iou
    corr_score = (spearmanr(y_true, y_pred)[0] + kendalltau(y_true, y_pred)[0])*0.25 + 0.5
    
    score = 0.8*iou_score + 0.2*corr_score

    return round(score, 5)


def get_question_ids(g):
    game_id = g['game_id'].values[0]
    questions_mask = np.array(list(g['questions'].values))
    question_ids = list(map(lambda x: list( 1000*game_id + np.where(x >  0)[0] + 1) + 
                                      list(-1000*game_id - np.where(x == 0)[0] - 1), questions_mask))
    return question_ids


def prepare_sparse_data(data):
    data = data[['game_id', 'team_id', 'members', 'questions']].copy()
    data['win_questions'] = data.groupby('game_id').apply(get_question_ids).explode().values
    data = data.drop(['game_id', 'questions'], axis=1).reset_index(drop=True)
    data.columns = ['team_id', 'player_id', 'question_id']
    data = data.explode('player_id').explode('question_id')
    teams = data['team_id'].values
    target = (data['question_id'] > 0).astype(int).values
    players = data['player_id'].values
    data['question_id'] = abs(data['question_id'])

    player_map = {pid: idx for idx, pid in enumerate(data['player_id'].unique())}
    total_players = len(player_map)
    question_map = {qid: idx + total_players for idx, qid in enumerate(data['question_id'].unique())}
    total_cols = len(player_map) + len(question_map)

    data['player_id'] = data['player_id'].map(player_map)
    data['question_id'] = data['question_id'].map(question_map)

    sparse_data = sparse.lil_matrix((len(data), total_cols), dtype=np.int8)
    sparse_data[range(len(data)), data['player_id']] = 1
    sparse_data[range(len(data)), data['question_id']] = 1

    del data

    player_inv_map = {idx:pid for pid,idx in player_map.items()}
    question_inv_map = {idx:qid for qid,idx in question_map.items()}

    return sparse_data, target, teams, players, player_inv_map, question_inv_map


def get_player_questioncount(data):
    data = data[['members', 'questions']].copy()
    data['win_count'] = data['questions'].apply(len)
    data = data[['members', 'win_count']]
    data = data.explode('members')
    data.columns = ['player_id', 'question_count']
    data['player_id'] = data['player_id'].astype(int)
    data = data.groupby('player_id').agg({'question_count': 'sum'}).reset_index()
    
    return data


def lr_decay(lr, n_iter):
    return lr * 10**(-4*n_iter/10**6)


def logloss(y, h, w):
    l = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return (w*l).mean()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class CustomLogisticRegression:
    def __init__(self, lr=0.01, num_iter=10, batch_size=1000, tol=0.00001, random_state=9):
        self.lr = lr
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y, weights=None):
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        n = X.shape[0]
        X = sparse.hstack((np.ones((n, 1)), X), format='csr')
        
        if self.coef_ is None:
            self.coef_ = np.random.normal(size=X.shape[1]) / n
        if weights is None:
            weights = np.ones(n)
            
        losses = []
        lr = self.lr

        for i in tqdm(range(self.num_iter)):
            batch_idxs = np.random.choice(n, self.batch_size)
            X_batch = X[batch_idxs, :]
            y_batch = y[batch_idxs]
            w_batch = weights[batch_idxs]
            z_batch = X_batch.dot(self.coef_)
            h_batch = sigmoid(z_batch)
            gradient = X_batch.T.dot(w_batch*(h_batch - y_batch)) / len(y_batch)
            lr = lr_decay(self.lr, i)
            self.coef_ -= lr * gradient
            
            loss = logloss(y_batch, h_batch, w_batch)
            losses.append(loss)
            if len(losses) == 100:
                mean_dloss = np.mean(np.abs(np.diff(losses)))
                losses = []
                if mean_dloss < self.tol:
                    print(f'Early convergence: lr={lr}')
                    break

    def predict(self, X):
        h = self.predict_proba(X)
        return (h >= 0.5).astype(int)
    
    def predict_proba(self, X):
        n = X.shape[0]
        X = sparse.hstack((np.ones((n, 1)), X))
        z = X.dot(self.coef_)
        h = sigmoid(z)
        return h


def plot_game_ratings_hist(game_ratings):
    school_ratings = game_ratings[game_ratings['name'].str.contains('Школ')]['game_rating'].values
    world_ratings  = game_ratings[game_ratings['name'].str.contains('Мир')]['game_rating'].values

    plt.gcf().set_size_inches((10, 6))
    plt.hist(game_ratings['game_rating'], bins=30, alpha=0.5)
    plt.hist(world_ratings, bins=3, color='red', label='Мировые')
    plt.hist(school_ratings, bins=3, color='green', label='Школьные')
    plt.title('Гистограмма сложности соревнований (больше=слсожнее)')
    plt.xlabel('Рейтинг соревнования')
    plt.legend()
    plt.show()

    
def plot_example():
    plt.gcf().set_size_inches(14, 5)

    x = np.arange(0, 1.001, 0.001)
    plt.plot(x, beta.pdf(a=3, b=2, x=x), label='игрок A (2 из 3)')
    plt.fill_between(x[:450], [0]*450, beta.pdf(a=3, b=2, x=x[:450]), alpha=0.5)
    plt.plot(x, beta.pdf(a=151, b=151, x=x), label='игрок B (150 из 300)')
    plt.fill_between(x[:480], [0]*480, beta.pdf(a=151, b=151, x=x[:480]), alpha=0.5)
    plt.legend()
    plt.show()