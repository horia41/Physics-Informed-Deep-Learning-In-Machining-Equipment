from taylor_loss import fit_taylor_constants, compute_set_T_ref, TaylorPhysicsLoss

constants = fit_taylor_constants(
    labels_csv = args.labels_csv,
    train_sets = [1, 2, 5, 7, 8, 10, 11],
)
set_T_ref = compute_set_T_ref(args.labels_csv)

physics_loss_fn = TaylorPhysicsLoss(
    constants     = constants,
    n_epochs      = 17,
    lambda_max    = 0.1,   # start here, tune later
    warmup_epochs = 5,
    set_T_ref     = set_T_ref,
)