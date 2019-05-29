module TwoStepNuc

using StatsBase
using Distributed
using ImageFiltering

const OFF4 = [CartesianIndex(1,0),CartesianIndex(-1,0),
              CartesianIndex(0,1),CartesianIndex(0,-1)]

const OFF5 = [CartesianIndex(0,0),
              CartesianIndex(1,0),CartesianIndex(-1,0),
              CartesianIndex(0,1),CartesianIndex(0,-1)]

LocationArray = Array{Int64,2}
ConcList = Array{Float64,1}
ConcArray = Array{Float64,2}
FillArray = Array{Bool,2}

struct KTAMParams
    gse::Float64
    eps::Float64
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


function flag_concs(tiletypes::Array{Char}, flagtiles::Array{Int64},
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


#lowlowgce
function lowlow_gce(concarray::ConcArray, depth::Int64, trials::Int64, p::KTAMParams)
    # set of critical nuclei:
    cns = Set{FillArray}()
    Gs = Float64[]
    minG = fill(Inf, size(concarray))

    for startloc in CartesianIndices(concarray)
            g, cn, tr, cnsize, formin = lowLowTrajectory_singlesite(startloc, p.gse, p.eps,
                                                                      concarray,
                                                                      depth; checkmp=true,
                                                                      cnar=cns)
            if formin < minG[startloc]
                minG[startloc] = formin
            end
            if g == Inf
                continue
            end
            # checkmp already ensures this
            #if cn in cns
            #    continue
            #end

            push!(Gs, g)
            push!(cns, cn)
    end
    Gce = -log(sum(exp.(-Gs)))
    return Gce, cns, Gs, minG
end

function probabilistic_gce(concarray::ConcArray, depth::Int64, trials::Int64, p::KTAMParams)
    # set of critical nuclei:
    cns = Set{FillArray}()
    Gs = Float64[]
    minG = fill(Inf, size(concarray))

    for startloc in CartesianIndices(concarray)
        for i in 1:trials
            g, cn, tr, cnsize, formin = probableTrajectory_singlesite(startloc, p.gse, p.eps,
                                                                      concarray,
                                                                      depth; checkmp=true,
                                                                      cnar=cns)
            if formin < minG[startloc]
                minG[startloc] = formin
            end
            if g == Inf
                continue
            end
            # checkmp already ensures this
            #if cn in cns
            #    continue
            #end

            push!(Gs, g)
            push!(cns, cn)
        end
    end
    Gce = -log(sum(exp.(-Gs)))
    return Gce, cns, Gs, minG
end



using StatsBase

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

using ProgressMeter

function pattern_match(tiletypes, usecodes, locmaps, images, params, concfun, boredmax;
                       initassign::Union{Nothing, Array{Int64,2}}=nothing, trialsperpoint=30,
                       depth=10, sn=1)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if initassign == nothing
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    else
        assignmap = initassign::Array{Int64,2}
    end
    
    println(assignmap)
    allplaceindices = CartesianIndices(assignmap)
    ntiles = length(tiletypes)
    
    function do_meas(am)
        cls = map( x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)),
                   eachindex(images) )
        return @distributed (max) for (i,cl) = cls
            gces = pmap(locmaps) do lm
                return probabilistic_gce(concs_to_array(lm, cl), depth, trialsperpoint,
                                         params)[1]
            end
            return gces[i]-minimum(gces[1:length(gces) .!= i])
        end
    end

    pr = Progress(boredmax, 1)
    bored = 0
    minmeas = do_meas(assignmap)
    println("Starting value: $minmeas")
    while bored < boredmax
        swap = sample(allplaceindices, 2*sn, replace=false)

        if all(assignmap[swap[1:sn]].==-1) && all(assignmap[swap[(sn+1):end]].==-1)
            continue
        end

        assignmap[swap[1:sn]], assignmap[swap[(sn+1):end]] = assignmap[swap[(sn+1):end]], assignmap[swap[1:sn]]

        meas = do_meas(assignmap)
        if meas < minmeas
            minmeas = meas
            println(assignmap)
            println("New value: $meas | b=$bored / $boredmax")
            flush(stdout)
            bored = 0
            pr = Progress(boredmax, 1)
        else
            assignmap[swap[1:sn]], assignmap[swap[(sn+1):end]] = assignmap[swap[(sn+1):end]], assignmap[swap[1:sn]]
            bored += 1
        end
        update!(pr, bored)
    end
    return assignmap, minmeas
    
end


using .TinyClimbers

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

function pm_meas_window_k2(am, images, ntiles, concfun, locmaps, kval)
    cls = map( x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)), eachindex(images) )
    function nfn(x)
        i, cl = x
        cas = [concs_to_array(lm, cl) for lm in locmaps]
        wvals = log.(sum(sum(sum(exp.(k^2*log.(imfilter(ca, centered(ones(k,k)/k^2), Inner()))))) for k=1:kval) for ca in cas)
        return -(wvals[i]-maximum(wvals[1:length(wvals) .!= i]))
    end
    #return @distributed (max) for x = cls
    #    return nfn(x)
    #end
    return mapreduce(nfn, max, cls)
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

function pm_meas_lowlow(am, images, ntiles, concfun, locmaps, depth, trialsperpoint, params)
    cls = map( x -> (x, assignmap_to_conclist(images[x], ntiles, am, concfun)),
               eachindex(images) )
    return @distributed (max) for (i,cl) = cls
        gces = pmap(locmaps) do lm
            return lowlow_gce(concs_to_array(lm, cl), depth, trialsperpoint,
                                     params)[1]
        end
        return gces[i]-minimum(gces[1:length(gces) .!= i])
    end
end

function pattern_match_TC(tiletypes, usecodes, locmaps, images, params, concfun;
                       initassign::Union{Nothing, Array{Int64,2}}=nothing, trialsperpoint=30,
                       depth=10, sn=1, nworkers=1)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if initassign == nothing
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    println(assignmap)
    ntiles = length(tiletypes)
    

    mr = TinyClimbers.MountainRange(x -> TinyClimbers.swapstep!(x, nsteps=sn),
                                    (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn),
                                    x -> pm_meas(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, assignmap, nworkers)

    return r
end

function pattern_match_TC_lowlow(tiletypes, usecodes, locmaps, images, params, concfun;
                       initassign::Union{Nothing, Array{Int64,2}}=nothing, trialsperpoint=30,
                       depth=10, sn=1, nworkers=1)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if initassign == nothing
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    println(assignmap)
    ntiles = length(tiletypes)
    

    mr = TinyClimbers.MountainRange(x -> TinyClimbers.swapstep!(x, nsteps=sn),
                                    (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn),
                                    x -> pm_meas_lowlow(x, images, ntiles, concfun, locmaps, depth, trialsperpoint, params))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, assignmap, nworkers)

    return r
end



function pattern_match_TC_window(tiletypes, usecodes, locmaps, images, params, concfun;
                       initassign::Union{Nothing, Array{Int64,2}}=nothing, kval=5,
                       sn=1, nworkers=1)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if initassign == nothing
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    println(assignmap)
    ntiles = length(tiletypes)
    

    mr = TinyClimbers.MountainRange(x -> TinyClimbers.swapstep!(x, nsteps=sn),
                                    (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn),
                                    x -> pm_meas_window_k2(x, images, ntiles, concfun, locmaps, kval))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, assignmap, nworkers)

    return r
end

              
function pattern_match_TC_window_new(tiletypes, usecodes, locmaps, images, params, concfun;
                                 initassign::Union{Nothing, Array{Int64,2}}=nothing, kval=5,
                                 sn=1, nworkers=1)
    allowedtiles = findall(x -> x in usecodes, tiletypes).-1
    if length(allowedtiles) > length(images[1])
        println("Oh dear")
    end
    if initassign == nothing
        assignmap = fill(-1, size(images[1]))
        initplaces = sample(CartesianIndices(assignmap), length(allowedtiles), replace=false)
        for i in eachindex(initplaces)
            assignmap[initplaces[i]] = allowedtiles[i]
        end
    else
        assignmap = initassign::Array{Int64,2}
    end

    println(assignmap)
    ntiles = length(tiletypes)
    

    mr = TinyClimbers.MountainRange(x -> TinyClimbers.swapstep!(x, nsteps=sn),
                                    (x,y) -> TinyClimbers.swapstep!(x, y, nsteps=sn),
                                    x -> pm_meas_window_new(x, images, ntiles, concfun, locmaps, kval))

    r = TinyClimbers.hillclimb_paranoidclimbers(mr, assignmap, nworkers)

    return r
end

"""
    N(m)

Return the number of tiles in assembly (1/0 values) `m`.
"""
function N(m)
    return sum(m)
end

function B(m)
    return sum(m[1:end-1,:] .& m[2:end,:]) + sum(m[:,1:end-1] .& m[:,2:end])
end

function zerolog(x)
    x==0 ? 0 : log(x)
end

function G(m, gse, ep, f)
    return (2*N(m)-B(m))*gse - N(m)*ep - sum(m .* zerolog.(f))
end



"""
    dGatt(m, gse, ep, f)

For an assembly `m`, and parameters, give the dG at each potential attachment 
site, with +Inf for any site that doesn't allow attachment, or already has a tile present.
"""
function calc_dGatt(m, gse, ep, f)
    r = similar(m, Float64)
    for ij in CartesianIndices(m)
        if m[ij] != 0
            r[ij] = Inf
        else
            b = (ij[1]>1 ? m[ij[1]-1,ij[2]] : 0) + (ij[2]>1 ? m[ij[1],ij[2]-1] : 0) +
                (ij[1]<size(m,1) ? m[ij[1]+1,ij[2]] : 0) + (ij[2]<size(m,2) ? m[ij[1],ij[2]+1] : 0)
            if b == 0
                r[ij] = Inf
            else
                r[ij] = (2 - b) * gse - ep - log(f[ij])
            end
        end
    end
    return r
end

function update_dGatt_and_probs_around!(dGatt, probs, loc, m, gse, ep, f)
    for offset in OFF5
        ij = loc + offset
        if !checkbounds(Bool, dGatt, ij)
            continue
        end        
        if @inbounds m[ij] != 0
            @inbounds dGatt[ij] = Inf
            @inbounds probs[ij] = 0
        else
            @inbounds b = (ij[1]>1 ? m[ij[1]-1,ij[2]] : 0) + (ij[2]>1 ? m[ij[1],ij[2]-1] : 0) +
                (ij[1]<size(m,1) ? m[ij[1]+1,ij[2]] : 0) + (ij[2]<size(m,2) ? m[ij[1],ij[2]+1] : 0)
            if b == 0
                @inbounds dGatt[ij] = Inf
            else
                @inbounds dGatt[ij] = (2 - b) * gse - ep - log(f[ij])
            end
            @inbounds probs[ij] = exp(-dGatt[ij])
        end
    end
    return dGatt, probs
end


function fillFavorable_sl!(m, loc, gse, ep, f, dGatt, probs)
    totaldG = 0
    while true
        didsomething = false
        for offset = OFF4
            ns = loc+offset
            if (!checkbounds(Bool, dGatt, ns)) 
                continue
            end
            if @inbounds dGatt[ns] < 0
                loc = ns
                @inbounds m[loc] = 1
                @inbounds totaldG += dGatt[ns]
                update_dGatt_and_probs_around!(dGatt, probs, ns, m, gse, ep, f)
                didsomething = true
                break
            end
        end
        if !didsomething
            break
        end
    end
    return m, totaldG
end

function fillFavorable(m, loc, gse, ep, f, dGatt)
    r = copy(m)
    g2 = copy(dGatt)
    return fillFavorable_sl!(r, loc, gse, ep, f, g2, similar(g2))
end


"""
    probableStep!(m, gse, ep, f)

Given an assembly `m` that *already has no favorable steps left* (not checked), of the
unfavorable steps, of all steps, take step i with probability e^{-Delta G_i} / (sum_i e^{-Delta G_i).  
Then add that step to `m`. Return `m` and the dG of the attachment.
"""
function probableStep!(m, gse, ep, f, dGatt, probs)
    totalprob = sum(probs)
    if totalprob == 0
        return m, 0
    end
    r = totalprob * rand()
    accum = 0
    loc = CartesianIndex(-1, -1)
    for ij in CartesianIndices(probs)
        accum += probs[ij]
        if accum >= r
            loc = ij
            break
        end
    end
    dG = dGatt[loc]
    m[loc] = 1
    update_dGatt_and_probs_around!(dGatt, probs, loc, m, gse, ep, f)
    return m, dG, dGatt, probs, loc
end

"""
    lowInitStep!(m, gse, ep, f)

Take the step that has the least dG.
"""
function lowInitStep!(m, gse, ep, f, dGatt)
    ij = argmin(dGatt)
    if dGatt[ij] == Inf
        return m, 0
    end
    m[ij] = 1
    return m, dGatt[ik], dGatt, loc
end

function lowInitLowFillStep(m, gse, ep, f, dGatt)
    r = similar(m)
    dGmin = Inf
    dGfilledmin = Inf
    loc = CartesianIndex(-1, -1)
    for ij in CartesianIndices(m)
        if dGatt[ij] == Inf
            dG = Inf
            continue
        end
        dG = dGatt[ij]
        if dG <= dGmin
            copyto!(r, m)
            r[ij] = 1
            mfilled, dGfilled = fillFavorable(r, ij, gse, ep, f, dGatt)
            if (dG < dGmin) || (dGfilled < dGfilledmin)
                dGfilledmin = dGfilled
                dGmin = dG
                loc = ij
            end
        end
    end
    if loc == CartesianIndex(-1, -1)
        return m, 0
    else
        copyto!(r,m)
        r[loc] = 1
        dG = dGatt[loc]
        dGatt = calc_dGatt(m, gse, ep, f)
        return r, dG, dGatt, loc
    end
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


function probableTrajectory_singlesite(loc, gse, ep, f, steps; checkmp=false,
                                       cnar=Set{Array{Bool,2}}())
    m = zeros(Bool, size(f))
    Acrit = zeros(Bool, size(f))
    Gcrit = -Inf
    Gtrace = zeros(steps)
    critstep = 1
    
    # Stop if there is no possible tile at the location
    if f[loc] == 0
        Gcrit = Inf
        return Gcrit, m, Gtrace, Inf, Inf
    end

    m[loc] = true
    Gtrace[1] = 2*gse - ep - log(f[loc])
    Gtrace[2] = Gtrace[1] # We prefill this, then modify it with the next step.
    dGatt = fill(Inf, size(m))
    probs = zeros(size(dGatt))
    update_dGatt_and_probs_around!(dGatt,probs,loc,m,gse,ep,f)
    for step = 2:steps
        m, dG, dGatt, probs, loc = probableStep!(m, gse, ep, f, dGatt, probs)
        Gtrace[step] += dG

        # mountain pass check
        if checkmp
            if m in cnar
                return Inf, m, Gtrace, Inf, Gtrace[step]
            end
        end        
        
        if Gtrace[step] > Gcrit
            Gcrit = Gtrace[step]
            critstep = step
            copyto!(Acrit, m)
        end
        m, dGfill = fillFavorable_sl!(m, loc, gse, ep, f, dGatt, probs)
        if step < steps
            Gtrace[step+1] = Gtrace[step] + dGfill # We prep the next site
        end
    end

    return Gcrit, Acrit, Gtrace, critstep, Gcrit
end

function lowLowTrajectory_singlesite(loc, gse, ep, f, steps; checkmp=false,
                                       cnar=Set{Array{Bool,2}}())
    m = zeros(Bool, size(f))
    Acrit = zeros(Bool, size(f))
    Gcrit = -Inf
    Gtrace = zeros(steps)
    critstep = 1
    
    # Stop if there is no possible tile at the location
    if f[loc] == 0
        Gcrit = Inf
        return Gcrit, m, Gtrace, Inf, Inf
    end

    m[loc] = true
    Gtrace[1] = 2*gse - ep - log(f[loc])
    Gtrace[2] = Gtrace[1] # We prefill this, then modify it with the next step.
    dGatt = fill(Inf, size(m))
    probs = zeros(size(dGatt))
    dGatt = calc_dGatt(m, gse, ep, f)

    for step = 2:steps
        m, dG, dGatt, loc = lowInitLowFillStep(m, gse, ep, f, dGatt)
        Gtrace[step] += dG

        # mountain pass check
        if checkmp
            if m in cnar
                return Inf, m, Gtrace, Inf, Gtrace[step]
            end
        end        
        
        if Gtrace[step] > Gcrit
            Gcrit = Gtrace[step]
            critstep = step
            copyto!(Acrit, m)
        end
        m, dGfill = fillFavorable_sl!(m, loc, gse, ep, f, dGatt, probs)
        if step < steps
            Gtrace[step+1] = Gtrace[step] + dGfill # We prep the next site
        end
    end

    return Gcrit, Acrit, Gtrace, critstep, Gcrit
end


end # module
