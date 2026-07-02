%% this loads in functional networks and does community detection on them
mice = ["mouse22"; "mouse25"; "mouse39"; "mouse35"; "mouse46"; "mouse51"; "mouse549"];
method="pearson_corr";

for m=1:length(mice)
    mouseID=mice{m};
    % designate the data directory
    drive = "./data/neural"

    if strcmp("mouse22", mouseID)
        days = ["042524", "042824", "042924", "043024", "050124"];
    elseif strcmp("mouse25", mouseID)
        days = ["042224", "042324", "042424", "042524"];
    elseif strcmp("mouse39", mouseID)
        days = ["042324", "042424", "042524", "042824"];
    elseif strcmp("mouse35", mouseID)
        days = ["070724", "070824", "070924", "071024"];
    elseif strcmp("mouse549", mouseID)
        days = ["081624", "081724", "081824", "081924"];
    elseif strcmp("mouse51", mouseID)
        days = ["081824", "081924", "082024", "082124"];
    elseif strcmp("mouse46", mouseID)
        days = ["081924", "082024", "082124", "082224"];
    else
        % nothing
    end

    mouse_dir = drive+"/"+mouseID+"/";
    calcium_data_path = mouse_dir + days(end);
    s2p_fld = calcium_data_path + "/";
    
    if strcmp("MI", method)
        active_graph = readNPY(s2p_fld + "active_FN_MI.npy");
	    empty_graph = readNPY(s2p_fld + "empty_FN_MI.npy");
    else
        active_graph = readNPY(s2p_fld + "active_FN_pearson_corr.npy");
	    empty_graph = readNPY(s2p_fld + "empty_FN_pearson_corr.npy");
    end
    % imshow(active_graph, [-1, 1])
    % imshow(empty_graph, [-1, 1])
    
    n_neurons = size(active_graph, 1);

    comp = active_graph-empty_graph;

    [ciu,Aall,anull,A,ciall] = get_FCmodules_MRCClite(comp,1000,10000,n_neurons,10^-5,10^5);
    [X,Y,indsort] = grid_communities(ciu);
    figure(m)
    imagesc(comp(indsort,indsort));
    save(s2p_fld+ "FN_sort_pearson.mat", "indsort")
end
