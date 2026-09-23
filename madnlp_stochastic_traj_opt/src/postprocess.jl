"Scrambled Sobol normal inputs, with separate reproducible initial/path streams."
function rollout_inputs(o::Options)
    qmc = QuasiMonteCarlo
    master = Xoshiro(o.monte_carlo_seed)
    initial_rng, path_rng, pairing_rng = (Xoshiro(s) for s in rand(master, UInt64, 3))
    function normals(dimension, rng)
        points = qmc.sample(o.monte_carlo_samples, dimension,
            qmc.SobolSample(R=qmc.MatousekScramble(base=2, pad=32, rng=rng)))
        # MatousekScramble includes the linear scramble and digital shift.
        # RNG streams are reproducible within Julia, but differ from NumPy.
        return quantile.(Normal(), clamp.(points, eps(Float64), 1-eps(Float64)))
    end
    initial = normals(NX, initial_rng)
    paths = normals(8o.n_arcs, path_rng)
    paths = paths[:, randperm(pairing_rng, o.monte_carlo_samples)]
    return (; initial, paths=reshape(paths, 8, o.n_arcs, o.monte_carlo_samples))
end

"Forward ShARK simulation with the Python sample-and-hold policy and norm saturation."
function monte_carlo_rollout(o::Options, Lc, sol, inputs; closed_loop=true, saturate=true)
    ns = size(inputs.initial,2)
    states = sol.means[:,1] .+ cholesky(Symmetric(o.x0_covariance)).L * inputs.initial
    trajectories = Array{Float64}(undef,NX,ns,o.n_arcs+1)
    trajectories[:,:,1] = states
    saturation_fraction = zeros(o.n_arcs)
    path_violation_fraction = zeros(2,o.n_arcs)
    for k in 1:o.n_arcs
        for (j,(a,b)) in enumerate(((o.a1,o.b1),(o.a2,o.b2)))
            path_violation_fraction[j,k] = count(>(b),vec(a'*states)) / ns
        end
        u = repeat(sol.feedforward[:,k],1,ns)
        closed_loop && (u .+= sol.gains[:,:,k]*(states .- sol.means[:,k]))
        magnitude = vec(sqrt.(sum(abs2,u;dims=1)))
        saturation_fraction[k] = count(>(o.u_max),magnitude) / ns
        if saturate
            u .*= reshape(min.(1.0,o.u_max ./ max.(magnitude,eps(Float64))),1,:)
        end
        for s in 1:ns
            states[:,s] = shark_step(states[:,s],u[:,s],sqrt(o.dt)*inputs.paths[1:4,k,s],
                sqrt(o.dt/12)*inputs.paths[5:8,k,s],Lc,o.dt)
        end
        trajectories[:,:,k+1] = states
    end
    covariances = cat((cov(trajectories[:,:,k];dims=2,corrected=true) for k in 1:o.n_arcs+1)...;dims=3)
    return (; trajectories, state_covariances=covariances, saturation_fraction, path_violation_fraction)
end

function ellipse_points(P; sigma=3.0, n=100)
    E = eigen(Symmetric(P))
    theta = range(0,2pi;length=n)
    return sigma * E.vectors * Diagonal(sqrt.(max.(E.values,0.0))) * [cos.(theta)'; sin.(theta)']
end

function plot_results(result, closed_loop, open_loop)
    # Off-screen rendering: no windows are opened during CLI runs.
    get!(ENV,"GKSwstype","100")
    @eval import Plots
    Base.invokelatest(_plot_results,result,closed_loop,open_loop)
end

function _plot_results(result, closed_loop, open_loop)
    o, sol = result.options, result.solution
    plt = Plots
    common = (; size=(800,700), dpi=160, fontfamily="Computer Modern",
        framestyle=:box, gridalpha=0.15, legend=:outertop, legendcolumns=3)
    function trajectory_plot(P, mc, label)
        p = plt.plot(;common...,xlabel="Position x",ylabel="Position y",aspect_ratio=:equal,title=label)
        xx = range(-1,12;length=200)
        for (j,(a,b)) in enumerate(((o.a1,o.b1),(o.a2,o.b2)))
            if !iszero(a[2])
                yy=(b .- a[1]*xx)/a[2]
                plt.plot!(p,xx,yy;color=:black,linestyle=:dash,linewidth=1,label="Path constraint $j")
            end
        end
        if mc !== nothing
            plt.scatter!(p,mc.trajectories[1,:,end],mc.trajectories[2,:,end];
                markersize=1,markerstrokewidth=0,alpha=0.15,color=:gray,label="Terminal samples")
        end
        plt.plot!(p,sol.means[1,:],sol.means[2,:];color=:black,marker=:cross,linewidth=1.5,label="Mean")
        for k in 1:o.n_arcs+1
            e=ellipse_points(P[1:2,1:2,k]) .+ sol.means[1:2,k]
            color=k==1 ? :blue : k==o.n_arcs+1 ? :red : :gray
            label=k==1 ? "Initial 3sigma" : k==o.n_arcs+1 ? "Terminal 3sigma" : ""
            plt.plot!(p,e[1,:],e[2,:];color,linewidth=0.8,label)
        end
        target=ellipse_points(o.xf_covariance[1:2,1:2]) .+ o.xf_mean[1:2]
        plt.plot!(p,target[1,:],target[2,:];color=:green,linestyle=:dash,label="Target 3sigma")
        px=sol.means[1,:]; py=sol.means[2,:]
        plt.xlims!(p,min(0.0,minimum(px)-2),max(11.0,maximum(px)+2))
        plt.ylims!(p,min(0.0,minimum(py)-2),max(8.0,maximum(py)+2))
        return p
    end
    p=trajectory_plot(sol.state_covariances,closed_loop,"Closed-loop covariance steering")
    plt.savefig(p,joinpath(o.output_dir,"closed_loop_traj.png"))
    if open_loop !== nothing
        p=trajectory_plot(open_loop.state_covariances,open_loop,"Open-loop rollout")
        plt.savefig(p,joinpath(o.output_dir,"open_loop_traj.png"))
    end
    t=o.dt*collect(0:o.n_arcs)
    magnitude=vec(sqrt.(sum(abs2,sol.feedforward;dims=1)))
    p=plt.plot(t,[magnitude;magnitude[end]];common...,seriestype=:steppost,
        xlabel="Time [s]",ylabel="Control norm",color=:black,label="Feedforward")
    plt.plot!(p,t,[sol.control_chance;sol.control_chance[end]];seriestype=:steppost,
        color=:blue,label="Chance bound")
    plt.hline!(p,[o.u_max];color=:red,linestyle=:dash,label="Control limit")
    plt.savefig(p,joinpath(o.output_dir,"control_magnitude.png"))
    p=plt.plot(sol.feedforward[1,:],sol.feedforward[2,:];common...,color=:black,
        marker=:cross,xlabel="Control x",ylabel="Control y",aspect_ratio=:equal,label="Feedforward")
    for k in 1:o.n_arcs
        e=ellipse_points(sol.control_covariances[:,:,k]) .+ sol.feedforward[:,k]
        plt.plot!(p,e[1,:],e[2,:];color=:gray,linewidth=0.8,label=k==1 ? "Control 3sigma" : "")
    end
    theta=range(0,2pi;length=200)
    plt.plot!(p,o.u_max*cos.(theta),o.u_max*sin.(theta);color=:red,linestyle=:dash,label="Control limit")
    plt.savefig(p,joinpath(o.output_dir,"control_covariance.png"))
    return nothing
end

"Write the solution/report and perform the Monte Carlo and plotting options."
function save_results(result)
    o, sol = result.options, result.solution
    mkpath(o.output_dir)
    closed_loop, open_loop = nothing,nothing
    if o.run_monte_carlo
        println("Verifying with $(o.monte_carlo_samples) scrambled Sobol paths ...")
        inputs=rollout_inputs(o)
        closed_loop=monte_carlo_rollout(o,result.Lc,sol,inputs;closed_loop=true)
        open_loop=monte_carlo_rollout(o,result.Lc,sol,inputs;closed_loop=false)
        result.report["monte_carlo_samples"]=o.monte_carlo_samples
        result.report["monte_carlo_seed"]=o.monte_carlo_seed
        result.report["closed_loop_terminal_variances"]=diag(closed_loop.state_covariances[:,:,end])
        result.report["peak_saturation_fraction"]=maximum(closed_loop.saturation_fraction)
        result.report["peak_path_violation_fraction"]=maximum(closed_loop.path_violation_fraction)
    end
    # Match the Python NPZ axis order exactly for the shared array names.
    arrays=Dict("means"=>sol.means,"feedforward"=>sol.feedforward,
        "gains"=>permutedims(sol.gains,(3,1,2)),
        "state_covariances"=>permutedims(sol.state_covariances,(3,1,2)),
        "control_covariances"=>permutedims(sol.control_covariances,(3,1,2)),
        "cholesky_factor"=>permutedims(sol.cholesky_factor),
        "terminal_margin"=>sol.terminal_margin,"Qc"=>result.Qc,"Lc"=>result.Lc)
    NPZ.npzwrite(joinpath(o.output_dir,"solutions.npz"),arrays)
    if closed_loop !== nothing
        NPZ.npzwrite(joinpath(o.output_dir,"monte_carlo.npz"),Dict(
            "closed_loop_trajectories"=>permutedims(closed_loop.trajectories,(3,1,2)),
            "open_loop_trajectories"=>permutedims(open_loop.trajectories,(3,1,2)),
            "closed_loop_covariances"=>permutedims(closed_loop.state_covariances,(3,1,2)),
            "open_loop_covariances"=>permutedims(open_loop.state_covariances,(3,1,2))))
    end
    open(joinpath(o.output_dir,"diagnostics.toml"),"w") do io
        TOML.print(io,result.report;sorted=true)
    end
    if o.make_plots
        plot_results(result,closed_loop,open_loop)
    end
    println("Results saved to ",abspath(o.output_dir))
    return nothing
end
