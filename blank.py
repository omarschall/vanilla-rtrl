#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:42:06 2020

@author: omarschall
"""

    #fig1.savefig(os.path.join(figs_path, 'fig22.pdf'), dpi=300, format='pdf')

    #checkpoints = {k:result[k] for k in result.keys() if 'checkpoint' in k}
    
    ### come up with graph metric that uses the alignment
    # do output gradient norms spike after topological changes
    # cross correlograms of different statistics
    # correct by magnitude of error signal
    # come up with statistical summary tests that actually test hypothesises
    # might need more interesting task if we can only test rudimentary hypohessis with it
    # do different algorithms have diff
    # just before a phase transition, i have an expectated gradient direction and variance
    # how often do you actually get to this state transition? is it mean or flunctuations
    # maybe phase transitions are actually smooth, geometry isn't acutally affected that much
    # for a gradient of the same size, do we get bigger fluctuations in network dynamics than i would expec
    # during a phse transition?
    # how do we make this conversation scale-invariant, both in terms of gradient norm
    # and norm of network dynamics geomoetrical changes
    

    #grad_norms = np.array(grad_norms)

    # for i_checkpoint in range(4900, 5000, 10):
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     plot_input_dependent_topology(checkpoint, i_input=None)

    # indices = list(range(10000, 50000, 1000)) + [80000, 200000, 500000, 999000]
    # indices = list(range(0, 10000, 10))
    
    # for i_checkpoint in range(8000, 8100, 10):
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     ref_checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint - 10)]
    #     #plot_input_dependent_topology(checkpoint, i_input=None)
    #     fig, ax = plt.subplots(2, 2)
    #     ax[0, 0].imshow(ref_checkpoint['adjmat_input_0'])
    #     ax[0, 1].imshow(checkpoint['adjmat_input_0'])
    #     ax[1, 0].imshow(ref_checkpoint['forwardshared_adjmat_input_0'])
    #     ax[1, 1].imshow(checkpoint['backshared_adjmat_input_0'])

    # for i_checkpoint in indices:
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     fig = plot_input_dependent_topology(checkpoint, i_input=None, return_fig=True)
    #     fig.savefig(os.path.join(figs_path, 'Fig_{}.pdf'.format(i_checkpoint)), dpi=300, format='pdf')
        
        
    
    # with open(os.path.join(figs_path, 'tex_tex.txt'), 'w') as f:
    #     for i_index in range(0, len(indices), 4):
    #         i_checkpoints = [indices[i_index],
    #                           indices[i_index + 1],
    #                           indices[i_index + 2],
    #                           indices[i_index + 3]]
    #         f.write('\\begin{figure}[h]\n' +
    #                 '\\center\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[0]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[1]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[2]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}\n'.format(i_checkpoints[3]) + 
    #                 '\\end{figure}\n')
     
    # for i_checkpoint in indices:
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     fig = plot_input_dependent_topology(checkpoint, i_input=None, return_fig=True)
    #     fig.savefig(os.path.join(figs_path, 'Fig_{}.pdf'.format(i_checkpoint)), dpi=300, format='pdf')
        
    #put_topolgoies_in_tex_script(indices, figs_path)

    # meeting notes
    # kep track of node diff penalty as separate metric
    # do cross corr properly
    # comparison of learning rates
    # vary magnitude of inputs
        
    # only really one solution to flip flop task
    # problem with two different solutions? will different learning algs
    # find diff solutions? which stages of learning matter most if you switch
    # algs?
    # compute hessian exactly to assess local minima
    
        # assess robustness to slightly different task configs
    # study REINFORCE or simlar things even as it fails
    # what are computational limitations of different bio plausible learning algorithms
    # if stochastic gradients not  learning, what do metrics or kinetic energy
    # or whatever reveal about their shortcomings
    # can topological space show what makes one algorithm better or worse than
    # another?
    # at asymptoptic performance, do gradient directions matter for preserving
    # topology or is it just about their norms being small?
    # can otherwise identical topologies be compared w.r.t. robustness to perturbations?
    # perturbations both to weight confirugartions and also input magnitudes.
    # lesioning neurons?
    # shallow minima, find basins of attraction that are robust. Are topologoical
    # strucutres signifiers of robustness?
    
    #make sure effective learning rates are comparable between algorithms
    #on aveage, what is the alignment with the "proper" gradient to get a measurement
    # of actual learning rate
    #with high dimsnioanl space and stochasicity, can ahve effectively much lower learning rate
    #potential argument about classes of learning algorithms best for learning tasks with
    # slow/fixed point structure andor memeory
    #look at offsets for aligning rflo and KE gradients, get snapshots of these values
    # and do proper correlation
    
    # need more mixing of algorithms, maybe stop at checkpoints along way,
    # try stupider algorithm
    # perturbation vs. error-based
    # pitch both as potential models for biolgical learning
    # stability problems in messing up good solution vs. notbeing able to find
    # solution to begin with
    # find checkpoint where RFLO alrady stabilizes, run REINFORCE for rest
    # of run


    #could just be variance of gradients kicking you out of small basins of
    #attraction in parameter space
    
    #different biological approximations suitable for different tasks
    #1-1 comparison of with and without noise within RFLO, both input noise
    #and intrinsic noise
    #knobs for signal to noise of inputs and of intrinsic noise
    #flip flop task with delays could be a mix of the two tasks
    
    #pitch: different learning algorithms have differetn capabilities of learning
    #"dynamical" systems or "memory"-based tasks
    # robustness of representation with 
    #REINFORCE doesn't learn memory stuff
    #noise in forward pass

    #figs_path = '/Users/omarschall/weekly-reports/report_08-19-2020/figs'
    #ssa_2.fig.savefig(os.path.join(figs_path, 'fig16.pdf'), dpi=300, format='pdf')

    # with open(os.path.join('saved_runs', file_name), 'wb') as f:
    #     pickle.dump(sim, f)
    
    # N_train = 50000
    # N_test = 5000
    # data = task.gen_data(N_train, N_test)
    # #sigma_ = np.sqrt(24 * 3) * sigma
    # learn_alg = REINFORCE(rnn, sigma=sigma, L2_reg=0.001, L1_reg=0.001, decay=0.1)
    # #learn_alg = Random_Noise_Gradients(rnn, sigma=0.007)
    # #optimizer = SGD_Momentum(lr=0.001, mu=0.6, clip_norm=None)
    # optimizer = Stochastic_Gradient_Descent(lr=0.005)
    
    # comp_algs = [RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg = 0.0001)]
    # monitors = ['alignment_matrix']
    
    # noisy_sim = Simulation(rnn)
    # noisy_sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
    #               sigma=sigma,
    #               comp_algs=comp_algs,
    #               monitors=monitors,
    #               verbose=True,
    #               report_accuracy=False,
    #               report_loss=True,
    #               checkpoint_interval=checkpoint_interval)
    
    # with open(os.path.join('saved_runs', '{}_{}'.format(name, 'REIN')), 'wb') as f:
    #     pickle.dump(noisy_sim, f)
    
    # if True:
    #     sim_ = Simulation(rnn)
    #     sim_ = linearly_interpolate_checkpoints(sim_,
    #                                             start_checkpoint=sim.checkpoints[0],
    #                                             end_checkpoint=sim.checkpoints[99000],
    #                                             density=100)
        

        
    # with open(os.path.join('saved_runs', 'interp_RFLO'), 'wb') as f:
    #     pickle.dump(sim_, f)

# for i_checkpoint in range(0, 300, 10):
#     checkpoint = sim.checkpoints[i_checkpoint]
#     analyze_checkpoint(checkpoint, data, verbose=False,
#                        sigma_pert=0.5, N=200, parallelize=True, n_PCs=2,
#                        N_iters=3000, same_LR_criterion=2000)
#     get_graph_structure(checkpoint)
    #train_VAE(checkpoint, big_data, T=10, latent_dim=256, lr=0.001)

# with open('notebooks/good_ones/clean_result_attempt', 'wb') as f:
#     pickle.dump(sim, f)
# with open('notebooks/good_ones/clean_result_attempt', 'rb') as f:
#     sim = pickle.load(f)
    
# big_data = task.gen_data(100, 500000)

# if False:
#     result = {}
    
#     for i_checkpoint in range(0, 200000, 10000):
#         # train_VAE(sim.checkpoints[i_checkpoint], big_data, T=10,
#         #           latent_dim=256, lr=0.001)
        
#         analyze_checkpoint(sim.checkpoints[i_checkpoint], data, verbose=False,
#                         sigma_pert=0.5, N=600, parallelize=False,
#                         N_iters=8000, same_LR_criterion=7000)
        
#         get_graph_structure(sim.checkpoints[i_checkpoint], parallelize=False)
        
#         result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])



#     checkpoint = sim.checkpoints[N_train - 1]
    

#a_init = np.random.normal(0, 0.01, rnn.a.shape)
# a_init = checkpoint['A_init'][2]
# rnn.reset_network(a_init=a_init)
# result = find_KE_minimum(rnn, LR=1, N_iters=20000, same_LR_criterion=15000,
#                          return_whole_optimization=True,
#                          return_period=20)
# analyze_checkpoint(checkpoint, data, verbose=False,
#                     sigma_pert=0.5, N=200, parallelize=False, n_PCs=3,
#                     N_iters=5000, same_LR_criterion=4000)
# get_graph_structure(checkpoint, N=200)



# train_VAE(sim.checkpoints[499999], big_data, T=10, latent_dim=128, lr=0.001)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(result['a_trajectory'][0], result['a_trajectory'][1], result['a_trajectory'][2])

# ssa = plot_checkpoint_results(checkpoint, data, plot_test_points=True,
#                               plot_cluster_means=False,
#                               plot_graph_structure=False)

# ssa.plot_in_state_space(result['a_trajectory'], mark_start_and_end=True, color='C1')

# data_path = '/scratch/oem214/vanilla-rtrl/library/vae_train'
# def process_all_results(data_path, N_jobs, rflo_checkpoints={}):
    
#     for i in range(N_jobs):

#         file_path = os.path.join(data_path, 'result_{}'.format(i))
    
#         try:
#             with open(file_path, 'rb') as f:
#                 result = pickle.load(f)
#         except FileNotFoundError:
#             continue
            
#         rflo_checkpoints.update(result)
            
#     return rflo_checkpoints


# print('processing results...')
# rflo_checkpoints = process_all_results(data_path, 200, rflo_checkpoints={})

# i_rflo = sorted([int(k.split('_')[-1]) for k in rflo_checkpoints.keys() if 'checkpoint' in k])
# checkpoints = [rflo_checkpoints['checkpoint_{}'.format(i_checkpoint)] for i_checkpoint in i_rflo]

# vae_distances = np.zeros((len(checkpoints), len(checkpoints)))
# pc_distances = np.zeros((len(checkpoints), len(checkpoints)))
# graph_distances = np.zeros((len(checkpoints), len(checkpoints)))
# for i in range(len(checkpoints)):
    
#     if i % 10 == 0:
#         with open('/scratch/oem214/vanilla-rtrl/log/vae_test.log', 'a') as f:
#             f.write('Calculating distnace row {}\n'.format(i))
    
#     for j in range(0, 6):
    
#         try:
#             checkpoint_1 = checkpoints[i]
#             checkpoint_2 = checkpoints[i + j]
#         except IndexError:
#             continue

#     #    rnn = checkpoint_1['rnn']
#     #    test_sim = Simulation(rnn)
#     #    test_sim.run(data,
#     #                  mode='test',
#     #                  monitors=['rnn.loss_'],
#     #                  verbose=False)
#     #    
#     #    losses.append(test_sim.mons['rnn.loss_'].mean())
#         vae_distances[i, i + j] = test_vae(model_checkpoint=checkpoint_1, data=big_data,
#                                             test_checkpoint=checkpoint_2) / 100000
        
#         pc_distances[i, i + j] = np.sum(checkpoint_1['V'] * checkpoint_2['V'])
#         graph_distances[i, i + j] = netsimile(checkpoint_1['adjacency_matrix'],
#                                               checkpoint_2['adjacency_matrix'])
        
# result['vae_distances'] = vae_distances
# result['pc_distances'] = pc_distances
# result['graph_distsances'] = graph_distances
#plt.figure()
#plt.imshow(distances)

# with open('notebooks/good_ones/sparsity_friend', 'rb') as f:
#     sim = pickle.load(f)
# result = {}
# for i_checkpoint in range(params['i_start'], params['i_start'] + 1000, 100):
#     analyze_checkpoint(sim.checkpoints[i_checkpoint], data, verbose=False,
#                         sigma_pert=0.5, N=600, parallelize=False,
#                         N_iters=8000, same_LR_criterion=7000)

#     get_graph_structure(sim.checkpoints[i_checkpoint], parallelize=False)

#     result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])
# with open('notebooks/good_ones/{}_net'.format(params['algorithm']), 'rb') as f:
#     sim = pickle.load(f)

# sim.resume_sim_at_checkpoint(data, 99999, N=100001, checkpoint_interval=100)

### --- GET CHECKPOINTS --- ###

# rflo_checkpoints = {}
# bptt_checkpoints = {}
# N_jobs = 200
# for i in range(N_jobs):
#     try:
#         # with open('library/bptt_rflo/result_{}'.format(i_job), 'rb') as f:
#         #     old_result = pickle.load(f)
#         with open('/Users/omarschall/cluster_results/vanilla-rtrl/bptt_rflo/result_{}'.format(i), 'rb') as f:
#             old_result = pickle.load(f)
#     except FileNotFoundError:
#         continue
#
#     # if file_exists:
#     #     for i_checkpoint in range(params['i_start'],
#     #                               params['i_start'] + 1000, 100):
#     #         get_graph_structure(result['checkpoint_{}'.format(i_checkpoint)],
#     #                             N=100, time_steps=5, parallelize=False)
#
#     if old_result['config']['algorithm'] == 'RFLO':
#         rflo_checkpoints.update(old_result)
#
#     if old_result['config']['algorithm'] == 'E-BPTT':
#         bptt_checkpoints.update(old_result)
#
#
# i_rflo = sorted([int(k.split('_')[-1]) for k in rflo_checkpoints.keys() if 'checkpoint' in k])
# i_bptt = sorted([int(k.split('_')[-1]) for k in bptt_checkpoints.keys() if 'checkpoint' in k])
#
# ### --- ANALYZE CHECKPOINTS --- ###
#
# rflo_distances = []
# for i in i_rflo[:-1]:
#
#     try:
#         checkpoint_1 = rflo_checkpoints['checkpoint_{}'.format(i)]
#         checkpoint_2 = rflo_checkpoints['checkpoint_{}'.format(i + 100)]
#     except KeyError:
#         continue
#
#     distance = SVCCA_distance(checkpoint_1, checkpoint_2, data)
#     rflo_distances.append(distance)
#
# bptt_distances = []
# for i in i_bptt[:-1]:
#
#     try:
#         checkpoint_1 = bptt_checkpoints['checkpoint_{}'.format(i)]
#         checkpoint_2 = bptt_checkpoints['checkpoint_{}'.format(i + 100)]
#     except KeyError:
#         continue
#
#     distance = SVCCA_distance(checkpoint_1, checkpoint_2, data)
#     bptt_distances.append(distance)
#
# result = {'rflo_distances': rflo_distances,
#           'bptt_distances': bptt_distances}
# result = {}
# for i_checkpoint in range(params['i_start'], params['i_start'] + 1000, 100):
#     analyze_checkpoint(sim.checkpoints[i_checkpoint], data, verbose=False,
#                         sigma_pert=0.5, N=600, parallelize=False,
#                         N_iters=8000, same_LR_criterion=7000)

#     result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])

with open('notebooks/good_ones/current_fave', 'rb') as f:
    sim = pickle.load(f)

# with open('/Users/omarschall/cluster_results/vanilla-rtrl/rflo_bptt/result_0', 'rb') as f:
#     result = pickle.load(f)
#     sim = result['sim']
#     rnn = sim.rnn

sim.resume_sim_at_checkpoint(data, i_checkpoint=params['segment'], N=1000,
                             checkpoint_interval=50)

N_iters = 5000
same_LR_criterion = 3000

#for j in range(len(sim.checkpoints.keys())):
for i_checkpoint in range(params['segment'], params['segment'] + 1000, 50):
#for j in range(5):

    #i_checkpoint = [_ for _ in sim.checkpoints.keys()][j]
    print('Analyzing checkpoint {}...'.format(i_checkpoint))

    rnn = sim.checkpoints[i_checkpoint]['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)

    transform = Vanilla_PCA(sim.checkpoints[i_checkpoint], data)
    V = transform(np.eye(rnn.n_h))
    # ssa = State_Space_Analysis(sim.checkpoints[i_checkpoint], data, transform)
    # V = ssa.transform(np.eye(n_hidden))

    fixed_points, initial_states = find_KE_minima(sim.checkpoints[i_checkpoint], data, N=150,
                                                  PCs=None, sigma_pert=0, N_iters=N_iters, LR=1,
                                                  weak_input=0, parallelize=False,
                                                  verbose=False, same_LR_criterion=same_LR_criterion)

    # with open('notebooks/good_ones/current_fave_FPs', 'wb') as f:
        #pickle.dump(fixed_points, f)

    A = np.array([d['a_final'] for d in fixed_points])
    A_init = np.array(initial_states)
    KE = np.array([d['KE_final'] for d in fixed_points])

    dbscan = DBSCAN(eps=0.5)
    dbscan.fit(A)
    dbscan.labels_

    # A_eigs = []
    # for i in range(A.shape[0]):

    #     rnn.reset_network(a=A[i])
    #     a_J = rnn.get_a_jacobian(update=False)
    #     A_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
    # A_eigs = np.array(A_eigs)


    # ssa.clear_plot()
    # ssa.plot_in_state_space(test_sim.mons['rnn.a'][1000:], False, 'C0', '.', alpha=0.05)
    # ssa.plot_in_state_space(A[A_eigs>1], False, 'C1', '*', alpha=1)
    # ssa.plot_in_state_space(A[A_eigs<1], False, 'C2', '*', alpha=1)
    # ssa.plot_in_state_space(A_init, False, 'C3', 'x', alpha=1)

    cluster_idx = np.unique(dbscan.labels_)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    cluster_means = np.zeros((n_clusters, rnn.n_h))
    for i in np.unique(dbscan.labels_):

        if i == -1:
            color = 'k'
            continue
        else:
            color = 'C{}'.format(i+1)
            cluster_means[i] = A[dbscan.labels_ == i].mean(0)


        # ssa.plot_in_state_space(A[dbscan.labels_ == i], False, color,
        #                         '*', alpha=0.5)

    # ssa.plot_in_state_space(cluster_means, False, 'k', 'X')

    #Saev results
    sim.checkpoints[i_checkpoint]['fixed_points'] = A
    sim.checkpoints[i_checkpoint]['KE'] = KE
    sim.checkpoints[i_checkpoint]['cluster_means'] = cluster_means
    sim.checkpoints[i_checkpoint]['cluster_labels'] = dbscan.labels_
    sim.checkpoints[i_checkpoint]['V'] = V

    # cluster_eigs = []
    # for i in range(cluster_means.shape[0]):

    #     rnn.reset_network(a=cluster_means[i])
    #     a_J = rnn.get_a_jacobian(update=False)
    #     cluster_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
    # cluster_eigs = np.array(cluster_eigs)

    # plt.figure()
    # plt.hist(cluster_eigs)
    # plt.title('Checkpoint {}'.format(i_checkpoint))
    # for i in range(2, 6):
    #     rnn.reset_network(a=test_sim.mons['rnn.a'][i*100])
    #     result = find_KE_minimum(rnn, return_whole_optimization=True, verbose=True,
    #                               N_iters=N_iters, same_LR_criterion=same_LR_criterion)
    #     ssa.plot_in_state_space(result['a_trajectory'], True, 'C{}'.format(i), '-', alpha=1)


    # ssa.fig.suptitle('Checkpoint {}'.format(i_checkpoint))

    # with open('notebooks/good_ones/current_fave', 'wb') as f:
    #     pickle.dump(sim, f)

# all_means = []
# for i_checkpoint in range(12000, 13000, 50):

#     #i_checkpoint = [_ for _ in sim.checkpoints.keys()][j]
#     print('Analyzing checkpoint {}...'.format(i_checkpoint))

#     checkpoint = sim.checkpoints[i_checkpoint]

#     labels = checkpoint['cluster_labels']
#     FPs = checkpoint['fixed_points']
#     cluster_means = checkpoint['cluster_means']
#     all_means.append(cluster_means)


#     rnn = sim.checkpoints[i_checkpoint]['rnn']
#     test_sim = Simulation(rnn)
#     test_sim.run(data,
#                   mode='test',
#                   monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#                   verbose=False)

    # transform = partial(np.dot, b=checkpoint['V'])
    # ssa = State_Space_Analysis(checkpoint, data, transform=transform)
    # ssa.plot_in_state_space(all_means, False, 'k', 'X', alpha=0.3)

#     for i in np.unique(checkpoint['cluster_labels']):

#         if i == -1:
#             color = 'k'
#             mark = 'o'
#             continue
#         else:
#             color = 'C{}'.format(i+1)
#             mark = '*'


#         ssa.plot_in_state_space(FPs[labels == i], False, color,
#                                 mark, alpha=0.5)
#     ssa.plot_in_state_space(cluster_means, False, 'k', 'X', alpha=0.3)
#     ssa.fig.suptitle('Checkpoint {}'.format(i_checkpoint))
    # fig = plt.figure()
    # plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
    # #plt.plot(data['test']['X'][:, 0])
    # plt.plot(data['test']['Y'][:, 0])
    # plt.xlim([2000, 3000])
    # plt.title('Checkpoint {}'.format(i_checkpoint))

    # plt.figure()
    # plt.hist(np.log10(KE), bins=20, color='C1')
    # plt.title('Checkpoint {}'.format(i_checkpoint))
    # plt.figure()
    # plt.hist(A_eigs, bins=20, color='C2')
    # plt.title('Checkpoint {}'.format(i_checkpoint))
#ssa.fig

# rnn_copy = deepcopy(rnn)
# rnn_copy.reset_network(a=A[28]+np.random.normal(0,0.3,64))
# result = find_KE_minimum(rnn_copy, return_whole_optimization=True, verbose=True)

# A_eigs = []
# for i in range(A.shape[0]):

#     rnn.reset_network(a=A[i])
#     a_J = rnn.get_a_jacobian(update=False)
#     A_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
# A_eigs = np.array(A_eigs)
# test_sim = Simulation(rnn)
# test_sim.run(data,
#              mode='test',
#              monitors=['rnn.loss_'],
#              verbose=False)
# test_loss = np.mean(test_sim.mons['rnn.loss_'])
# processed_data = {'test_loss': test_loss}

#fixed_points = find_KE_minima(sim.checkpoints[-1], data, N=200, parallelize=True,
#                              N_iters=100000, verbose=True)

# test_sim = Simulation(rnn)
# test_sim.run(data,
#               mode='test',
#               monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#               verbose=False)

# rnn = sim.checkpoints[30000]['rnn']
# test_sim = Simulation(rnn)
# test_sim.run(data,
#              mode='test',
#              monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#              verbose=False)

# plt.figure()
# plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
# #plt.plot(data['test']['X'][:, 1])
# plt.plot(data['test']['Y'][:, 0])
# plt.xlim([0, 1000])

# plt.figure()
# plt.hist(np.log10(KE), bins=20)

if os.environ['HOME'] == '/Users/omarschall':
    params = {'mu': 0.8, 'clip_norm': 0.1, 'L2_reg': 0.001}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

#with open('notebooks/good_ones/another_try', 'rb') as f:
#    sim = pickle.load(f)
#
#sim.checkpoint_model()
#
#task = Flip_Flop_Task(3, 0.05)
#data = task.gen_data(1000, 8000)
#
#result = analyze_all_checkpoints(sim.checkpoints, find_KE_minima, data,
#                                 verbose_=True, N=20, N_iters=1000)

random_FPs = [np.random.normal(0, 1, (128)) for _ in range(8)]
noisy_points = np.array([random_FPs[np.random.randint(8)] + np.random.normal(0, 0.3, (128)) for _ in range(1000)])
random_FPs = np.array(random_FPs)
U, S, V = np.linalg.svd(noisy_points)
PCs = V[:,:3]
proj = random_FPs.dot(PCs)
noisy_proj = noisy_points.dot(PCs)
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.plot(proj[:,0], proj[:,1], proj[:,2], '.')
ax.plot(noisy_proj[:,0], noisy_proj[:,1], noisy_proj[:,2], '.', alpha=0.2)
for x in proj:
    for y in proj:

        ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color='C0', alpha=0.1)

task = Flip_Flop_Task(3, 0.05, tau_task=1)
data = task.gen_data(10000, 6000)

n_in = task.n_in
n_hidden = 128
n_out = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)

optimizer = SGD_Momentum(lr=0.001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
learn_alg = Efficient_BPTT(rnn, 10, L2_reg=params['L2_reg'])
#learn_alg = RFLO(rnn, alpha=alpha)
#learn_alg = Only_Output_Weights(rnn)
#learn_alg = RTRL(rnn, M_decay=0.7)
#learn_alg = RFLO(rnn, alpha=alpha)

comp_algs = []
monitors = ['learn_alg.rec_grads-norm']
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
optimizer = SGD_Momentum(lr=0.0001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
optimizer = SGD_Momentum(lr=0.00001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
sim.checkpoint_model()

fixed_points = find_KE_minima(sim.checkpoints[-1], data, N=20,
                              verbose=True, parallelize=True)

test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_'],
             verbose=False)
test_loss = np.mean(test_sim.mons['rnn.loss_'])
processed_data = {'test_loss': test_loss}

#plt.figure()
#x = configs_array['T_horizon']
#mean_results = results_array.mean(-1)
#ste_results = results_array.std(-1)/np.sqrt(20)
#for i in range(4):
#    col = 'C{}'.format(i)
#    mu = mean_results[:,i]
#    ste = ste_results[:, i]
#    plt.plot(x, mu, color=col)
#    plt.fill_between(x, mu - ste, mu + ste, alpha=0.3, color=col)
#plt.legend([str(lr) for lr in configs_array['LR']])
#plt.xticks(x)


# test_sim = Simulation(rnn)
# test_sim.run(data,
#              mode='test',
#              monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#              verbose=False)

# plt.figure()
# plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
# #plt.plot(data['test']['X'][:, 0])
# plt.plot(data['test']['Y'][:, 0])

# Load network
network_name = 'j_boxman'
with open(os.path.join('notebooks/good_ones', network_name), 'rb') as f:
    rnn = pickle.load(f)

task = Flip_Flop_Task(3, 0.05)
np.random.seed(0)
n_test = 10000
data = task.gen_data(0, n_test)
#test_sim = deepcopy(sim)
test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

find_slow_points_ = partial(find_slow_points, N_iters=10000, return_period=100,
                            N_seed_2=1)
#results = find_slow_points_([test_sim, 0, 0])
pool = mp.Pool(mp.cpu_count())
N_seed_1 = 8
results = pool.map(find_slow_points_, zip([test_sim]*N_seed_1,
                                          range(N_seed_1),
                                          [i_job]*N_seed_1))
pool.close()
A = [results[i][0] for i in range(N_seed_1)]
speeds = [results[i][1] for i in range(N_seed_1)]
LR_drop_times = [results[i][2] for i in range(N_seed_1)]
result = {'A': A, 'speeds': speeds}

task = Flip_Flop_Task(3, 0.05)
np.random.seed(0)
n_test = 10000
data = task.gen_data(0, n_test)
#test_sim = deepcopy(sim)
test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.002)
ssa.fig.axes[0].set_xlim([-0.6, 0.6])
ssa.fig.axes[0].set_ylim([-0.6, 0.6])
ssa.fig.axes[0].set_zlim([-0.8, 0.8])

data_path = '/Users/omarschall/cluster_results/vanilla-rtrl/slow_points_2'

all_speeds = []

for i_job in range(30):
    try:
        with open(os.path.join(data_path, 'result_{}'.format(i_job)), 'rb') as f:
            result = pickle.load(f)
            A = result['A']
            all_speeds.append(result['speeds'][-1])
    except FileNotFoundError:
        continue
    for i in range(20):
        col = 'C1'
        #ssa.plot_in_state_space(A[i][:-1,:], color=col)
        slowness = np.minimum(1/np.sqrt(result['speeds'][i][-1]), 4)
        ssa.plot_in_state_space(A[i][-1,:].reshape((1,-1)), 'x', color=col, alpha=0.3,
                                markersize=slowness)



#task = Flip_Flop_Task(3, 0.5)
#data = task.gen_data(100000, 5000)
#
#n_in = task.n_in
#n_hidden = 32
#n_out = task.n_out
#
#W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
#W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
#W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
#b_rec = np.zeros(n_hidden)
#b_out = np.zeros(n_out)
#
#alpha = 1
#
#rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
#          activation=tanh,
#          alpha=alpha,
#          output=identity,
#          loss=mean_squared_error)
#
#optimizer = Stochastic_Gradient_Descent(lr=0.001)
#SG_optimizer = Stochastic_Gradient_Descent(lr=0.001)
#
#if params['algorithm'] == 'Only_Output_Weights':
#    learn_alg = Only_Output_Weights(rnn)
#if params['algorithm'] == 'RTRL':
#    learn_alg = RTRL(rnn, L2_reg=0.01)
#if params['algorithm'] == 'UORO':
#    learn_alg = UORO(rnn)
#if params['algorithm'] == 'KF-RTRL':
#    learn_alg = KF_RTRL(rnn)
#if params['algorithm'] == 'R-KF-RTRL':
#    learn_alg = Reverse_KF_RTRL(rnn)
#if params['algorithm'] == 'BPTT':
#    learn_alg = Future_BPTT(rnn, params['T_horizon'])
#if params['algorithm'] == 'DNI':
#    learn_alg = DNI(rnn, SG_optimizer)
#if params['algorithm'] == 'DNIb':
#    J_lr = 0.001
#    learn_alg = DNI(rnn, SG_optimizer, use_approx_J=True, J_lr=J_lr,
#                    SG_label_activation=tanh, W_FB=W_FB)
#    learn_alg.name = 'DNIb'
#if params['algorithm'] == 'RFLO':
#    learn_alg = RFLO(rnn, alpha=alpha)
#if params['algorithm'] == 'KeRNL':
#    sigma_noise = 0.0000001
#    base_learning_rate = 0.01
#    kernl_lr = base_learning_rate/sigma_noise
#    KeRNL_optimizer = Stochastic_Gradient_Descent(kernl_lr)
#    learn_alg = KeRNL(rnn, KeRNL_optimizer, sigma_noise=sigma_noise)
#
#comp_algs = []
#monitors = []
#
#sim = Simulation(rnn)
#sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
#        comp_algs=comp_algs,
#        monitors=monitors,
#        verbose=True,
#        check_accuracy=False,
#        check_loss=True)