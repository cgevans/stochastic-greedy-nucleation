module TwoStepNuc

#export pm_meas, basic_concfun, pattern_match_TC_window_new, pattern_match_TC, KTAMParams

using StatsBase
using Distributed
using ImageFiltering
import TinyClimbers
using DataStructures
using Random
using TinyClimbers
const OFF4 = [CartesianIndex(1,0),CartesianIndex(-1,0),
              CartesianIndex(0,1),CartesianIndex(0,-1)]

const OFF5 = [CartesianIndex(0,0),
              CartesianIndex(1,0),CartesianIndex(-1,0),
              CartesianIndex(0,1),CartesianIndex(0,-1)]

const LocationArray = Array{Int64,2}
const ConcList = Array{Float64,1}
const ConcArray = Array{Float64,2}
const FillArray = Array{Bool,2}

struct KTAMParams
    gmc::Float64
    gse::Float64    
    alpha::Float64
    kf::Float64
end

G_se(p::KTAMParams) = p.gse
G_mc(p::KTAMParams) = p.gmc
k_f(p::KTAMParams) = p.kf
epsilon(p::KTAMParams) = 2*p.gse - p.gmc
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

function probabilistic_gce(concarray::ConcArray, trials::Int64, p::KTAMParams,
                           depth::Int64=typemax(Int64))
    # set of critical nuclei:
    found_cns = Array{FillArray,1}()
    found_Gcns = Float64[]
    found_traces = Array{Array{Float64,1},1}()
    min_G_per_site = fill(Inf, size(concarray))

    for i in 1:trials
        # Our starting site is chosen probabilistically based on concentration.
        starting_site = choose_position(concarray)

        # This will give us a new critical nucleus and trajectory, starting from
        # that site, or will fail, if it passes through a critical nucleus that
        # has already been found.
        Gcn, cn, G_trace, crit_step = probable_trajectory(concarray, starting_site, p,
                                                          found_cns,
                                                          depth)

        # Gcn is returned as Inf if the trajectory passed through a previously-
        # found critical nucleus.
        if Gcn == Inf
            continue
        end

        push!(found_Gcns, Gcn)
        push!(found_cns,  cn)
        push!(found_traces, G_trace)
    end

    Gce = -log(sum(exp.(-found_Gcns)))

    # We will sort all our critical nuclei, so that the
    # most likely ones are first.
    sortkey = sortperm(found_Gcns)
    
    return Gce, found_Gcns[sortkey], found_cns[sortkey], found_traces[sortkey]
end

function probable_trajectory(concmult, starting_site, p::KTAMParams,
                             previous_cns::Array{FillArray,1}, depth::Int64=typemax(Int64))

    state = zeros(Bool, size(concmult))

    current_cn = zeros(Bool, size(concmult))
    current_Gcn = -Inf
    current_critstep = 1

    # keeps track of whether the state could, by an arbitrary series
    # of additions, become previous_cns[i]
    could_become_previous_cns = ones(Bool, length(previous_cns))
    
    trace = Float64[]
    
    # Stop if the initial site has no tile there!
    if concmult[starting_site] == 0
        return Inf, current_cn, trace, current_critstep
    end

    # To start, add the initial tile to the state.
    state[starting_site] = true

    # We include alpha here, because not including it is
    # really confusing.  Every tile attachment doesn't include it,
    # because the attachment G_mc includes it, but here, since we
    # are just looking at a single tile, where alpha should not be
    # included, we need to compensate for it.
    # This means we have $[c]_{eq} = u_0 e^{-G}$, as we would expect,
    # for both the single tile, and all future assemblies.
    G = G_mc(p) - alpha(p) - log(concmult[starting_site])
    push!(trace, G)

    stepnum = 2

    dGatt = fill(Inf, size(concmult))
    probs = zeros(size(dGatt))
    update_dGatt_and_probs_around!(concmult, dGatt, probs,
                                   state, starting_site, p)

    while stepnum <= depth
        site, dG = probabilistic_step!(concmult, dGatt, probs,
                                     state, p)

        # loc = -1,-1 → no more steps were possible
        if site == CartesianIndex(-1,-1)
            break
        end

        # break if we are in a state that was in previous_cns.
        # But we'll be smart here.  There's no point in checking equality if
        # a previous state (we're like the aTAM here with no detachments) had a
        # tile in a place where the CN we're comparing with didn't.  So we'll
        # keep track of this, and save ourselves quite a bit of time.
        #if state in previous_cns
        #    return Inf, current_cn, trace, current_critstep
        #end

        for i in eachindex(could_become_previous_cns)
            if !could_become_previous_cns[i]
                continue
            elseif any( state .& (.! previous_cns[i]))
                could_become_previous_cns[i] = false
            elseif state == previous_cns[i]
                return Inf, current_cn, trace, current_critstep
            end
        end
        
                
        
        # we're actually in a new state.  Update stuff:
        G += dG
        push!(trace, G)

        # Are we in a state of higher G than we've seen before?
        # If so, that seems like our new best guess of critical
        # nucleus.
        if G > current_Gcn
            current_Gcn = G
            current_critstep = stepnum
            copyto!(current_cn, state)
        end
        
        # Fill the state with dG<0 steps.
        dG = fillFavorable_sl!(concmult, dGatt, probs,
                               state, site, p)
        G += dG
    end

    return current_Gcn, current_cn, trace, current_critstep
end

function probabilistic_step!(concmult, dGatt, probs, state, p)
    totalprob = sum(probs)
    if totalprob == 0
        return CartesianIndex(-1, -1), Inf  # -1, -1 indicates no possible addition
    end
    
    loc = choose_position_presummed(probs, totalprob)

    dG = dGatt[loc]
    state[loc] = true
    
    update_dGatt_and_probs_around!(concmult, dGatt, probs,
                                   state, loc, p)    

    return loc, dG
end

"""modifies dGatt and probs"""
function update_dGatt_and_probs_around!(concmult, dGatt, probs,
                                        state, loc, p)
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
            @inbounds b = (ij[1]>1 ? state[ij[1]-1,ij[2]] : 0) + (ij[2]>1 ? state[ij[1],ij[2]-1] : 0) +
                (ij[1]<size(state,1) ? state[ij[1]+1,ij[2]] : 0) + (ij[2]<size(state,2) ? state[ij[1],ij[2]+1] : 0)
            if b == 0
                @inbounds dGatt[ij] = Inf
                @inbounds probs[ij] = 0
            else
                @inbounds dGatt[ij] = G_mc(p) - b*G_se(p) - log(concmult[ij])
                @inbounds probs[ij] = exp(-dGatt[ij])
            end
        end
    end
    return dGatt, probs
end

function fillFavorable_sl!(concmult, dGatt, probs,
                           state, loc, p)
    totaldG = 0
    while true
        didsomething = false
        # The only place we might have new dG<0 steps is
        # in sites adjacent to previous attachments, so
        # we check there.
        for offset = OFF4
            trialsite = loc + offset

            # We may be out of bounds
            if (!checkbounds(Bool, dGatt, trialsite)) 
                continue
            end

            # Do we have a dG < 0 site?
            if @inbounds (dG = dGatt[trialsite]) < 0
                loc = trialsite
                @inbounds state[loc] = true
                @inbounds totaldG += dG
                update_dGatt_and_probs_around!(concmult, dGatt, probs,
                                               state, loc, p)    
                didsomething = true
                break
            end
        end
        if !didsomething
            break
        end
    end
    return totaldG
end

function N(m)
    return sum(m)
end

function B(m)
    return sum(m[1:end-1,:] .& m[2:end,:]) + sum(m[:,1:end-1] .& m[:,2:end])
end

function G(concmult, state, p)
    return N(state)*G_mc(p) - B(state)*G_se(p) - sum(log.(concmult[state])) - alpha(p)
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
    typearray = fill('0', maxT+1)
    for tile in alltiles
        n = parse(Int64, tile[2:end])
        typearray[n+1] = tile[1]
    end
    if '0' in typearray
        println("Something is wrong!")
    end
    return typearray
end

function flag_concs(tiletypes::Array{Char,1}, flagtiles::Array{Int64,1},
                    depletetypes::Array{Char}, baseconc::Float64,
                    flagconc::Float64, deplete::Float64)
    # Tiles start from 0, as in SHAM, so tiletypes[tilenum+1] = type of tilenum
    concarray = fill(baseconc, size(tiletypes))
    for tn in flagtiles
        concarray[tn+1] = flagconc
    end
    
    for ti in eachindex(tiletypes)
        if tiletypes[ti] in depletetypes
            concarray[ti] -= deplete
        end
    end

    return concarray
end

function flag_concs(tiletypes, flagtiles::Array{String},
                    depletetypes, baseconc,
                    flagconc, deplete)
    flagtiles_int = map(x -> parse(Int64, x[2:end]), flagtiles)
    return flag_concs(tiletypes, flagtiles_int, depletetypes, baseconc, flagconc, deplete)
end

function make_hillflag(tiletypes, flagcodes, locmaps, nflagtiles, params, flagmult, boredmax)
    # Goal here is to use a hill-climbing algorithm just to try to make
    # a nice flag, using only certain code tiles.

    allowedplaces = findall(x-> x in flagcodes, tiletypes).-1
    conclist = ones(Float64, size(tiletypes))
    places = sample(allowedplaces, nflagtiles, replace=false)

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
    return 1.0 + ival*10.0
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
    cls = map( x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)), eachindex(images) )
    function nfn(x)
        i, cl = x
        gces = map(locmaps) do lm
            return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint,
                                     params)[1]
        end
        gces[i]-minimum(gces[1:length(gces) .!= i])
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

function flag_meas_twostep_new(alloweds, nhigh, chigh, ntiles, locmaps, depth, trialsperpoint, params)
    cl = ones(ntiles)
    cl[alloweds[1:nhigh]].=chigh
    gces = map(locmaps) do lm
        return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint,
                                 params)[1]
    end
    return gces[1]-minimum(gces[2:end])
end

function conc_meas_twostep(concs, nhigh, chigh, ntiles, locmaps, depth, trialsperpoint, params; retall=false)
    gces = map(locmaps) do lm
        return probabilistic_gce(concs_to_array(lm, concs), depth, trialsperpoint,
                                 params)[1]
    end
    if !retall
        return gces[1]-minimum(gces[2:end])
    else
        return gces[1]-minimum(gces[2:end]), gces
    end
    
end

function pm_meas_window_new(am, images, ntiles, concfun, locmaps, kval)

    cls = map( x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)), eachindex(images) )

    function nfn(x)
        i, cl = x
        cas = [log.(concs_to_array(lm, cl)) for lm in locmaps]

        wvals = log.(sum(sum(exp.(kval^2*imfilter(ca, centered(ones(kval,kval)/kval^2), Inner())))) for ca in cas)
        return -(wvals[i]-maximum(wvals[1:length(wvals) .!= i]))
    end
    #return @distributed (max) for x = cls
    #    return nfn(x)
    #end
    return mapreduce(nfn, max, cls)
end

function flag_meas_window_new(alloweds, nhigh, chigh, ntiles, locmaps, kval)
    cl = ones(ntiles)
    cl[alloweds[1:nhigh]].=chigh
    cas = [log.(concs_to_array(lm, cl)) for lm in locmaps]
    wvals = log.(sum(sum(exp.(kval^2*imfilter(ca, centered(ones(kval,kval)/kval^2), Inner())))) for ca in cas)
    return -(wvals[1]-maximum(wvals[2:end]))
end

function pattern_match_TC(tiletypes, usecodes, locmaps, images, params, concfun;
                       initassign::Union{Nothing, Array{Int64,2}}=nothing, trialsperpoint=30,
                       depth=10, sn=1, pids=[2], rightblank=false)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if (initassign == nothing) && (!rightblank)
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    elseif (initassign == nothing) && rightblank
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
        sf = x -> TinyClimbers.swapstep!(x, nsteps=sn)
        sb = (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn)
        meas = x -> pm_meas(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    else    
        sf = swap_fix_m1!
        sb = doswap! 
        meas = x -> pm_meas(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    end        
    
    
    r = TinyClimbers.hillclimb_paranoidclimbers(meas, sf, sb, assignmap, pids)

    return r
end
              
function pattern_match_TC_window_new(tiletypes, usecodes, locmaps, images, concfun;
                                 initassign::Union{Nothing, Array{Int64,2}}=nothing, kval=5,
                                 sn=1, pids=[2], rightblank=false, uit=1000)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if (initassign == nothing) && (!rightblank)
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    elseif (initassign == nothing) && rightblank
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
        sf = x -> TinyClimbers.swapstep!(x, nsteps=sn)
        sb = (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn)
        meas = x -> pm_meas_window_new(x, images, ntiles, concfun, locmaps, kval)
    else
        sf = swap_fix_m1!
        sb = doswap!
        meas = x -> pm_meas_window_new(x, images, ntiles, concfun, locmaps, kval)
    end        

    r = TinyClimbers.hillclimb_paranoidclimbers(meas, sf, sb, assignmap, pids, uit=uit)

    return r
end

function restricted_swap!(value::Array, indexpool::Array)
    n = sample(indexpool, 2, replace=false)
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
              
function makeflag_fastmodel(tiletypes, usecodes, locmaps, nhigh, chigh;
                            initallowed::Union{Nothing, Array{Int64,1}}=nothing,
                            kval=5, sn=1, nworkers=1)

    if initallowed== nothing
        initallowed = findall(x -> x in usecodes, tiletypes)
        shuffle!(initallowed)
    end
    al = copy(initallowed) # copy to avoid modifying input initcl

    ntiles = length(tiletypes)
    
    mr = TinyClimbers.MountainRange(x -> partswap!(x, nhigh),
                                    (x,y) -> doswap!(x, y),
                                    x -> flag_meas_window_new(x, nhigh, chigh, ntiles, locmaps, kval))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, al, nworkers)

    return r
end

function makeflag_goodmodel(tiletypes, usecodes, locmaps, nhigh, chigh, params;
                            initallowed::Union{Nothing, Array{Int64,1}}=nothing,
                            kval=5, sn=1, nworkers=1, trialsperpoint=300, depth=14)

    if initallowed== nothing
        initallowed = findall(x -> x in usecodes, tiletypes)
        shuffle!(initallowed)
    end
    al = copy(initallowed) # copy to avoid modifying input initcl

    ntiles = length(tiletypes)
    
    mr = TinyClimbers.MountainRange(x -> partswap!(x, nhigh),
                                    (x,y) -> doswap!(x, y),
                                    x -> flag_meas_twostep_new(x, nhigh, chigh, ntiles, locmaps, depth, trialsperpoint, params))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, al, nworkers)

    return r
end

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
