module TwoStepNuc

#export pm_meas, basic_concfun, pattern_match_TC_window_new, pattern_match_TC, KTAMParams

using StatsBase
using Distributed
using ImageFiltering
import TinyClimbers
import SHAM
using DataStructures
using Random
using GroupSlices
const OFF4 = [
    CartesianIndex(1, 0),
    CartesianIndex(-1, 0),
    CartesianIndex(0, 1),
    CartesianIndex(0, -1),
]

const OFF5 = [
    CartesianIndex(0, 0),
    CartesianIndex(1, 0),
    CartesianIndex(-1, 0),
    CartesianIndex(0, 1),
    CartesianIndex(0, -1),
]

const LocationArray = Array{Int64,2}
const ConcList = Array{Float64,1}
const ConcArray = Array{Float64,2}
const FillArray = BitArray

struct KTAMParams
    gmc::Float64
    gse::Float64
    alpha::Float64
    kf::Float64
end

G_se(p::KTAMParams) = p.gse
G_mc(p::KTAMParams) = p.gmc
k_f(p::KTAMParams) = p.kf
epsilon(p::KTAMParams) = 2 * p.gse - p.gmc
alpha(p::KTAMParams) = p.alpha
kh_f(p::KTAMParams) = p.kf * exp(-alpha)

""" 
Choose a starting site, given a concentration array.  Note that this could be
concentration factors, or actual concentrations: the calculation should be the same.

We're just choosing here based on total concentration, not dimer formation rate,
which could be the better option but probably just makes things more complicated.
"""
function choose_position_presummed(weights, summed)
    trigger = summed * rand()
    accum = 0
    for i in eachindex(weights)
        @inbounds accum += weights[i]
        if accum >= trigger
            return CartesianIndices(weights)[i]
        end
    end
end

function choose_position(weights)
    full_val = sum(weights)
    return choose_position_presummed(weights, full_val)
end


"""
Run the SGM for fixed parameters and concentration multiple array, for a fixed
number of trials.
"""
function probabilistic_gce(
    farray::ConcArray,
    trials::Int64,
    p::KTAMParams,
    depth::Int64 = typemax(Int64);
    calc_weightsize_G = false,
    store_assemblies = false
)
    maxsize = sum(farray .> 0)

    # set of critical nuclei:
    found_cns = Array{FillArray,1}()
    found_Gcns = Float64[]
    found_traces = Array{Array{Float64,1},1}()
    size_traces = Array{Array{Int, 1}, 1}()
    nucrates = Float64[]
    min_Gcn_per_site = fill(Inf, size(farray))
    weight_Gcn_per_site = fill(Inf, size(farray))    
    nucrate_per_site = fill(NaN, size(farray))    
    min_size_per_site = fill(Inf, size(farray))
    weight_size_per_site = fill(Inf, size(farray))
    sized_assemblies = [[] for _ in range(1, maxsize)]
    sized_assembly_Gs = [[] for _ in range(1, maxsize)]
    assembly_traces = []

    stoppedtrials = 0
    goodtrials = 0
    finished_trials = 0

    # Find the final G
    finalG = G(farray, farray .> 0, p)
    baseG = G_mc(p) - alpha(p)

    return_assemblies = calc_weightsize_G | store_assemblies

    while finished_trials < trials
        # Our starting site is chosen probabilistically based on concentration.
        starting_site = choose_position(farray)
        # This will give us a new critical nucleus and trajectory, starting from
        # that site, or will fail, if it passes through a critical nucleus that
        # has already been found.

        out = probable_trajectory(farray, starting_site, p, depth, store_assemblies=return_assemblies)
        
        finished_trials += 1

        # We store, if we are storing, sized assemblies regardless of whether we found
        # a new critical nucleus.
        if calc_weightsize_G
            for k in range(1, length(out.assemblies))
                if out.assemblies[k] in sized_assemblies[k]
                    continue
                end
                push!(sized_assemblies[size_trace[k]], out.assemblies[k])
                push!(sized_assembly_Gs[size_trace[k]], out.G_trace[k])
            end
        end

        push!(found_Gcns, out.Gcn)
        push!(found_cns, out.cn)
        push!(found_traces, out.G_trace)
        push!(nucrates, out.nucrate)
        push!(size_traces, out.size_trace)
        if store_assemblies
            push!(assembly_traces, out.assemblies)
        end
        goodtrials += 1
    end

    # Check for duplicate critical nuclei, and remove them.
    inds = firstinds(groupslices(found_cns))
    found_Gcns = found_Gcns[inds]
    found_cns = found_cns[inds]
    found_traces = found_traces[inds]
    size_traces = size_traces[inds]
    nucrates = nucrates[inds]
    if store_assemblies
        assembly_traces = assembly_traces[inds]
    end
    

    if length(found_Gcns) > 0
        Gce = -log(sum(exp.(-found_Gcns)))
    else
        Gce = Inf
    end

    # We will sort all our critical nuclei, so that the
    # most likely ones are first.
    sortkey = sortperm(found_Gcns)

    sizes = found_cns.|>sum
    
    for ij in CartesianIndices(min_Gcn_per_site)
        containing_cns = [ i for i in 1:length(found_cns) if found_cns[i][ij] ]
        if length(containing_cns) == 0
            continue
        end
        
        gcnc = found_Gcns[containing_cns]
        min_Gcn_per_site[ij] = minimum(gcnc)
        weight_Gcn_per_site[ij] = sum(gcnc .* exp.(-gcnc)) / sum(exp.(-gcnc))
        nucrate_per_site[ij] = sum( nucrates[containing_cns] ./ sizes[containing_cns] )
        min_size_per_site[ij] = minimum(sizes[containing_cns])
        weight_size_per_site[ij] = sum(sizes[containing_cns] .* exp.(-gcnc)) / sum(exp.(-gcnc))
    end

    if length(sizes) > 0
        min_size = minimum(sizes)   
        weight_size = sum(sizes .* exp.(-found_Gcns)) / sum(exp.(-found_Gcns))
    else
        min_size = 1000
        weight_size = 1000.0
    end
    
    weightsizeG = fill(NaN, maxsize)
    numsized = fill(NaN, maxsize)

    if calc_weightsize_G
        for i in range(1, maxsize)
            numsized[i] = sized_assembly_Gs[i] |> length
            if numsized[i] == 0
                continue
            end
            Gs = sized_assembly_Gs[i]
            Z = sum(exp.(-Gs))
            weightsizeG[i] = sum(Gs .* exp.(-Gs)) / Z
        end
    end
    
    return (
        gce = Gce,
        gcns = found_Gcns[sortkey],
        nucrate = sum(nucrates[sortkey]),
        nucrates = nucrates[sortkey],
        cns = found_cns[sortkey],
        traces = found_traces[sortkey],
        size_traces = size_traces[sortkey],
        stoppedpct = stoppedtrials / trials,
        ncns = length(found_cns),
        min_Gcn_per_site = min_Gcn_per_site,
        weight_Gcn_per_site = weight_Gcn_per_site,
        min_size_per_site = min_size_per_site,
        weight_size_per_site = weight_size_per_site,
        nucrate_per_site = nucrate_per_site,
        min_size = min_size,
        weight_size = weight_size,
        finalG = finalG,
        baseG = baseG,
        num_per_size = numsized,
        G_weighted_per_size = weightsizeG,
        assembly_traces = store_assemblies ? assembly_traces[sortkey] : []
    )
end

"""
Calculate a single SGM trajectory, for a concentration multiple array `farray`, starting from position
`starting_site`, with parameters `p`, going up to `max_steps` steps.  If `store_assemblies`, then also
return all the assemblies along the trajectory.
"""
function probable_trajectory(
    farray::ConcArray,
    starting_site,
    p::KTAMParams,
    max_steps::Int64 = typemax(Int64);
    store_assemblies::Bool = false
)

    state = falses(size(farray))

    current_cn = falses(size(farray))
    current_Gcn = -Inf
    current_critstep = 1
    current_frate = 0

    assemblies = []
    
    trace = Float64[]
    size_trace = Int64[]

    # Stop if the initial site has no tile there!
    if farray[starting_site] == 0
        return (Gcn = Inf, cn = current_cn, G_trace = trace, crit_step = current_critstep, size_trace = size_trace, nucrate = current_frate, assemblies=assemblies)
    end

    # To start, add the initial tile to the state.
    state[starting_site] = true
    statesize = 1

    # We include alpha here, because not including it is
    # really confusing.  Every tile attachment doesn't include it,
    # because the attachment G_mc includes it, but here, since we
    # are just looking at a single tile, where alpha should not be
    # included, we need to compensate for it.
    # This means we have $[c]_{eq} = u_0 e^{-G}$, as we would expect,
    # for both the single tile, and all future assemblies.
    G = G_mc(p) - alpha(p) - log(farray[starting_site])
    push!(trace, G)
    push!(size_trace, statesize) 

    stepnum = 2

    dGatt = fill(Inf, size(farray))
    probs = zeros(size(dGatt))
    update_dGatt_and_probs_around!(farray, dGatt, probs, state, starting_site, p)

    while stepnum <= max_steps
        site, dG = probabilistic_step!(farray, dGatt, probs, state, p)
        statesize += 1
        # loc = -1,-1 → no more steps were possible
        # With vast depth, this is now the standard way of ending.
        if site == CartesianIndex(-1, -1)
            break
        end
        

        # we're actually in a new state.  Update stuff:
        oldG = G
        pdG = dG
        G += dG
        push!(trace, G)
        push!(size_trace, statesize)
        if store_assemblies
            push!(assemblies, copy(state))
        end

        # Are we in a state of higher G than we've seen before?
        # If so, that seems like our new best guess of critical
        # nucleus.
        if G > current_Gcn
            current_Gcn = G
            current_critstep = stepnum
            copyto!(current_cn, state)
            is_current_critnuc = true
        else
            is_current_critnuc = false
        end

        # Fill the state with dG<0 steps.
        dG, frate, ntiles = fillFavorable_sl!(farray, dGatt, probs, state, site, p, is_current_critnuc)
        statesize += ntiles
        if is_current_critnuc
            current_frate = frate * exp(-current_Gcn)
            dgsetot_of_prob = pdG - G_mc(p) - log(farray[site])
            rrate = k_f(p) * exp(alpha(p)) * exp(dgsetot_of_prob)
        end
        G += dG
    end

    # We are now a the end of the trajectory.

    return (Gcn = current_Gcn, 
            cn = current_cn, 
            G_trace = trace, 
            crit_step = current_critstep, 
            size_trace = size_trace,
            nucrate = current_frate,
            assemblies = assemblies)
end

function probabilistic_step!(farray::Array{Float64, 2}, dGatt::Array{Float64, 2}, probs::Array{Float64, 2}, state::BitArray{2}, p::KTAMParams)
    totalprob = sum(probs)
    if totalprob == 0
        return CartesianIndex(-1, -1), Inf  # -1, -1 indicates no possible addition
    end

    loc = choose_position_presummed(probs, totalprob)

    dG = dGatt[loc]
    state[loc] = true

    update_dGatt_and_probs_around!(farray, dGatt, probs, state, loc, p)

    return loc, dG
end

"""modifies dGatt and probs"""
function update_dGatt_and_probs_around!(concmult, dGatt, probs, state, loc, p)
    # We need to update probabilities for the location, and
    # the adjacent cells.
    for offset in OFF5
        ij = loc + offset

        # no need to update stuff outside the boundary!
        # after this, we can assume the location is inbounds.
        if !checkbounds(Bool, dGatt, ij)
            continue
        end

        if @inbounds state[ij] != 0   # we never revisit a filled site
            @inbounds dGatt[ij] = Inf
            @inbounds probs[ij] = 0
        else
            # Calculate the number of bonds that would be made by attaching a tile at ij.
            @inbounds b =
                (ij[1] > 1 ? state[ij[1]-1, ij[2]] : 0) +
                (ij[2] > 1 ? state[ij[1], ij[2]-1] : 0) +
                (ij[1] < size(state, 1) ? state[ij[1]+1, ij[2]] : 0) +
                (ij[2] < size(state, 2) ? state[ij[1], ij[2]+1] : 0)
            if b == 0
                @inbounds dGatt[ij] = Inf
                @inbounds probs[ij] = 0
            else
                @inbounds dGatt[ij] = G_mc(p) - b * G_se(p) - log(concmult[ij])
                @inbounds probs[ij] = exp(-dGatt[ij])
            end
        end
    end
end

function fillFavorable_sl!(concmult, dGatt, probs, state, loc, p, is_current_critnuc::Bool)
    totaldG = 0

    # Note: dGatt and probs must be current here!  They are updated by probabilistic_step! .

    # Potential forward sites have dGatt < 0.  They can only be adjacent to the location.
    frate = 0
    if is_current_critnuc
        for offset in OFF4
            trialsite = loc + offset            
            if (!checkbounds(Bool, dGatt, trialsite))
                continue
            end
            if dGatt[trialsite] < 0
                frate += k_f(p) * exp(-G_mc(p) + alpha(p))
            end
        end
    end

    ntiles = 0
    while true
        didsomething = false
        # The only place we might have new dG<0 steps is
        # in sites adjacent to previous attachments, so
        # we check there.
        for offset in OFF4
            trialsite = loc + offset

            # We may be out of bounds
            if (!checkbounds(Bool, dGatt, trialsite))
                continue
            end

            # Do we have a dG < 0 site?
            if @inbounds (dG = dGatt[trialsite]) < 0
                loc = trialsite
                @inbounds state[loc] = true
                ntiles += 1
                @inbounds totaldG += dG
                update_dGatt_and_probs_around!(concmult, dGatt, probs, state, loc, p)
                didsomething = true
                break
            end
        end
        if !didsomething
            break
        end
    end
    return totaldG, frate, ntiles
end

function N(m)
    return sum(m)
end

function B(m)
    return sum(m[1:end-1, :] .& m[2:end, :]) + sum(m[:, 1:end-1] .& m[:, 2:end])
end

function G(concmult, state, p)
    return N(state) * G_mc(p) - B(state) * G_se(p) - sum(log.(concmult[state])) - alpha(p)
end

"""
    concs_to_array(locs, conclist) -> concarray

Create a concentration array from a location array
and a concentration specification.

Note: here we have a problem with zero-indexing.  For consistency with
Python, tiles are zero-indexed.  -1 in the location array corresponds
with a blank space.  Index i in the conclist refers to tile i-1.
"""
function concs_to_array(locs::LocationArray, conclist::ConcList)
    concarray = similar(locs, Float64)
    @simd for i in eachindex(locs)
        @inbounds x = locs[i]
        if x == -1
            @inbounds concarray[i] = 0
        else
            concarray[i] = conclist[x+1]
        end
    end
    return concarray
end

"""
    process_cmap(cmap) -> locarray

Create a location array from a compact map.

Note that you probably need to use permutedims first.
"""
function process_cmap(cmap)
    cmi = similar(cmap, Int64)
    @simd for i in eachindex(cmap)
        x = cmap[i]
        if x == "."
            cmi[i] = -1
        else
            cmi[i] = parse(Int64, x[2:end])
        end
    end
    return cmi
end

function maketypes(cmaps...)
    alltiles = Set{String}()
    maxT = 0
    for map in cmaps
        for x in map
            if x == "."
                continue
            else
                push!(alltiles, x)
                n = parse(Int64, x[2:end])
                if n > maxT
                    maxT = n
                end
            end
        end
    end
    typearray = fill('0', maxT + 1)
    for tile in alltiles
        n = parse(Int64, tile[2:end])
        typearray[n+1] = tile[1]
    end
    if '0' in typearray
        println("Something is wrong!")
    end
    return typearray
end

function make_hillflag(
    tiletypes,
    flagcodes,
    locmaps,
    nflagtiles,
    params,
    flagmult,
    boredmax,
)
    # Goal here is to use a hill-climbing algorithm just to try to make
    # a nice flag, using only certain code tiles.

    allowedplaces = findall(x -> x in flagcodes, tiletypes) .- 1
    conclist = ones(Float64, size(tiletypes))
    places = sample(allowedplaces, nflagtiles, replace = false)

    @simd for i in places
        conclist[i+1] = flagmult
    end


    function do_meas(cl)
        gces = pmap(locmaps) do lm
            return probabilistic_gce(concs_to_array(lm, cl), 10, 10, params)[1]
        end
        return gces[1] - minimum(gces[2:end])
    end

    bored = 0
    minmeas = Inf
    while bored < boredmax
        newp = rand(allowedplaces)
        if newp in places
            continue
        end
        oldi = rand(1:size(places)[1])
        oldp = places[oldi]
        conclist[oldp+1] = 1.0
        conclist[newp+1] = flagmult
        meas = do_meas(conclist)
        if meas < minmeas
            minmeas = meas
            places[oldi] = newp
            println("New value: $meas | b=$bored / $boredmax")
            flush(stdout)
            bored = 0
        else
            conclist[oldp+1] = flagmult
            conclist[newp+1] = 1.0
            bored += 1
        end
    end
    return conclist, places, minmeas
end

function basic_concfun(ival)
    return 1.0 + ival * 10.0
end

function assignmap_to_conclist(image, ntiles, assignmap, concfun)
    concarray = ones(Float64, ntiles)
    for i in CartesianIndices(assignmap)
        if assignmap[i] == -1
            continue
        end
        concarray[assignmap[i]+1] = concfun(image[i])
    end
    return concarray
end

function pm_meas(am, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    cls = map(
        x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)),
        eachindex(images),
    )
    function nfn(x)
        i, cl = x
        gces = map(locmaps) do lm
            return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint, params)[1]
        end
        gces[i] - minimum(gces[1:length(gces).!=i])
    end
    return mapreduce(nfn, max, cls)
    #return @distributed (max) for (i,cl) = cls
    #    gces = pmap(locmaps) do lm
    #        return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint,
    #                                 params)[1]
    #    end
    #    return gces[i]-minimum(gces[1:length(gces) .!= i])
    #end
end

function flag_meas_twostep_new(
    alloweds,
    nhigh,
    chigh,
    ntiles,
    locmaps,
    depth,
    trialsperpoint,
    params,
)
    cl = ones(ntiles)
    cl[alloweds[1:nhigh]] .= chigh
    gces = map(locmaps) do lm
        return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint, params)[1]
    end
    return gces[1] - minimum(gces[2:end])
end

function conc_meas_twostep(
    concs,
    nhigh,
    chigh,
    ntiles,
    locmaps,
    depth,
    trialsperpoint,
    params;
    retall = false,
)
    gces = map(locmaps) do lm
        return probabilistic_gce(concs_to_array(lm, concs), depth, trialsperpoint, params)[1]
    end
    if !retall
        return gces[1] - minimum(gces[2:end])
    else
        return gces[1] - minimum(gces[2:end]), gces
    end

end

function pm_meas_window_new(am, images, ntiles, concfun, locmaps, kval)

    cls = map(
        x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)),
        eachindex(images),
    )

    function nfn(x)
        i, cl = x
        cas = [log.(concs_to_array(lm, cl)) for lm in locmaps]

        wvals =
            log.(
                sum(sum(exp.(
                    kval^2 * imfilter(ca, centered(ones(kval, kval) / kval^2), Inner()),
                ))) for ca in cas
            )
        return -(wvals[i] - maximum(wvals[1:length(wvals).!=i]))
    end
    #return @distributed (max) for x = cls
    #    return nfn(x)
    #end
    return mapreduce(nfn, max, cls)
end

function flag_meas_window_new(alloweds, nhigh, chigh, ntiles, locmaps, kval)
    cl = ones(ntiles)
    cl[alloweds[1:nhigh]] .= chigh
    cas = [log.(concs_to_array(lm, cl)) for lm in locmaps]
    wvals =
        log.(
            sum(sum(exp.(
                kval^2 * imfilter(ca, centered(ones(kval, kval) / kval^2), Inner()),
            ))) for ca in cas
        )
    return -(wvals[1] - maximum(wvals[2:end]))
end

function pattern_match_TC(
    tiletypes,
    usecodes,
    locmaps,
    images,
    params,
    concfun;
    initassign::Union{Nothing,Array{Int64,2}} = nothing,
    trialsperpoint = 30,
    depth = 10,
    sn = 1,
    pids = [2],
    rightblank = false,
)
    allowedtiles = findall(x -> x in usecodes, tiletypes) .- 1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if (initassign === nothing) && (!rightblank)
        assignmap = fill(-1, size(images[1]))
        initplaces =
            sample(CartesianIndices(assignmap), length(allowedtiles), replace = false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    elseif (initassign === nothing) && rightblank
        assignmap = fill(-1, size(images[1]))
        shuffle!(allowedtiles)
        for (at, ix) in zip(allowedtiles, CartesianIndices(assignmap))
            assignmap[ix] = at
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    ntiles = length(tiletypes)


    if !rightblank
        sf = x -> TinyClimbers.swapstep!(x, nsteps = sn)
        sb = (x, y) -> TinyClimbers.swapstep!(x, y, nsteps = sn)
        meas =
            x -> pm_meas(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    else
        sf = swap_fix_m1!
        sb = doswap!
        meas =
            x -> pm_meas(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    end


    r = TinyClimbers.hillclimb_paranoidclimbers(meas, sf, sb, assignmap, pids)

    return r
end

function pattern_match_TC_window_new(
    tiletypes,
    usecodes,
    locmaps,
    images,
    concfun;
    initassign::Union{Nothing,Array{Int64,2}} = nothing,
    kval = 5,
    sn = 1,
    pids = [2],
    rightblank = false,
    uit = 1000,
)
    allowedtiles = findall(x -> x in usecodes, tiletypes) .- 1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if (initassign === nothing) && (!rightblank)
        assignmap = fill(-1, size(images[1]))
        initplaces =
            sample(CartesianIndices(assignmap), length(allowedtiles), replace = false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    elseif (initassign === nothing) && rightblank
        assignmap = fill(-1, size(images[1]))
        shuffle!(allowedtiles)
        for (at, ix) in zip(allowedtiles, CartesianIndices(assignmap))
            assignmap[ix] = at
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    ntiles = length(tiletypes)

    if !rightblank
        sf = x -> TinyClimbers.swapstep!(x, nsteps = sn)
        sb = (x, y) -> TinyClimbers.swapstep!(x, y, nsteps = sn)
        meas = x -> pm_meas_window_new(x, images, ntiles, concfun, locmaps, kval)
    else
        sf = swap_fix_m1!
        sb = doswap!
        meas = x -> pm_meas_window_new(x, images, ntiles, concfun, locmaps, kval)
    end

    r = TinyClimbers.hillclimb_paranoidclimbers(meas, sf, sb, assignmap, pids, uit = uit)

    return r
end

function restricted_swap!(value::Array, indexpool::Array)
    n = sample(indexpool, 2, replace = false)
    value[n[1]], value[n[2]] = value[n[2]], value[n[1]]
    return n
end

function doswap!(value::Array, n::Array)
    value[n[1]], value[n[2]] = value[n[2]], value[n[1]]
end

function partswap!(value::Array, divline::Int64)
    n1 = rand(1:divline)
    n2 = rand((divline+1):length(value))
    value[n1], value[n2] = value[n2], value[n1]
    return [n1, n2]
end

function swap_fix_m1!(value::Array)
    r = 1:(length(value))
    while true
        i1 = rand(r)
        i2 = rand(r)
        if (i2 == i1) || (value[i1] == -1) || (value[i2] == -1)
            continue
        end
        value[i1], value[i2] = value[i2], value[i1]
        return [i1, i2]
    end
end

# function makeflag_fastmodel(
#     tiletypes,
#     usecodes,
#     locmaps,
#     nhigh,
#     chigh;
#     initallowed::Union{Nothing,Array{Int64,1}} = nothing,
#     kval = 5,
#     sn = 1,
#     nworkers = 1,
# )

#     if initallowed === nothing
#         initallowed = findall(x -> x in usecodes, tiletypes)
#         shuffle!(initallowed)
#     end
#     al = copy(initallowed) # copy to avoid modifying input initcl

#     ntiles = length(tiletypes)

#     mr = TinyClimbers.MountainRange(
#         x -> partswap!(x, nhigh),
#         (x, y) -> doswap!(x, y),
#         x -> flag_meas_window_new(x, nhigh, chigh, ntiles, locmaps, kval),
#     )

#     r = TinyClimbers.hillclimb_paranoidclimbers(mr, al, nworkers)

#     return r
# end

# function makeflag_goodmodel(
#     tiletypes,
#     usecodes,
#     locmaps,
#     nhigh,
#     chigh,
#     params;
#     initallowed::Union{Nothing,Array{Int64,1}} = nothing,
#     kval = 5,
#     sn = 1,
#     nworkers = 1,
#     trialsperpoint = 300,
#     depth = 14,
# )

#     if initallowed === nothing
#         initallowed = findall(x -> x in usecodes, tiletypes)
#         shuffle!(initallowed)
#     end
#     al = copy(initallowed) # copy to avoid modifying input initcl

#     ntiles = length(tiletypes)

#     mr = TinyClimbers.MountainRange(
#         x -> partswap!(x, nhigh),
#         (x, y) -> doswap!(x, y),
#         x -> flag_meas_twostep_new(
#             x,
#             nhigh,
#             chigh,
#             ntiles,
#             locmaps,
#             depth,
#             trialsperpoint,
#             params,
#         ),
#     )

#     r = TinyClimbers.hillclimb_paranoidclimbers(mr, al, nworkers)

#     return r
# end

function makeConcMap(map, flagtiles, sharedcodes, flagconc, sharedconc, uniqueconc)
    f = similar(map, Float64)
    for i in eachindex(map)
        if map[i] == "."
            f[i] = 0
        elseif map[i] in flagtiles
            f[i] = flagconc
        elseif map[i][1] in sharedcodes
            f[i] = sharedconc
        else
            f[i] = uniqueconc
        end
    end
    return f
end

end # module
