import numpy as np
import stabiliser_code as sc
from ldpc import BpOsdDecoder
from galois import GF2
import networkx as nx
from tableau import tableau2paulistring



def create_error_probs_from_logical(num_qubits,logical,p_base:float=0.1,p_punish:float=0.02): #logical in (x|z) repr.

    if isinstance(logical,list):
        logical=sc.tableau_list_to_matrix(logical)

    if logical.shape[1]//2!=num_qubits:
        raise ValueError("Logical defined for {} qubits. But num_qubits is {}".format(logical.shape//2,num_qubits))

    channel_probs = np.full(3 * num_qubits, p_base) #note that this is in (x|z|y) notation
    logical=logical.ravel()

    try:
        lx = logical[:num_qubits].astype(bool)
        lz = logical[num_qubits:].astype(bool)
    except(TypeError):
        lx=np.array(logical[:num_qubits]).astype(bool)
        lz=np.array(logical[num_qubits:]).astype(bool)

    channel_probs[:num_qubits][lx]=p_punish #z error unlikely when logical has x entry, note swap of x and z for channel_probs
    channel_probs[num_qubits:2*num_qubits][lz]=p_punish
    channel_probs[2*num_qubits:][lx ^ lz]=p_punish

    return channel_probs



def find_short_first_logical(T,other_logical_repr,CSS:bool=False,max_iter:int=100): #TODO: what happens if result is None? This may happen?


    #decoder problem now takes on representation (x|z|x plus z)
    #when giving decoder matrix in (x|z|x+z) format and without Omega, the error that is given back is in
    #(z|x|x+z) format!
    #TODO

    if sc.rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')

    n_qubits=T.shape[1]//2 #integer division

    T=sc.to_gf2_tableau(T)

    if isinstance(other_logical_repr,list):
        other_logical_repr=sc.tableau_list_to_matrix(other_logical_repr)

    if CSS:
        print("css")
        pass

    T=sc.append_logical_to_tableau(T,other_logical_repr)
    T=sc.turn_tableau_into_TXZY(T)

    syndrome = np.zeros(T.shape[0], dtype=int)
    syndrome[-1]=1 #assumes logical qubit is last row
    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)


    if CSS:
        pass

    else:
        channel_probs = np.full(3 * n_qubits, 0.1)

        decoder = BpOsdDecoder(T, channel_probs=channel_probs,
                                max_iter=max_iter, bp_method="ms",
                                osd_method="osd_cs", osd_order=6)
        result = decoder.decode(syndrome_1d)

        if result is None or not np.asarray(result).any():
            return None
        else:
            assert np.array_equal((T @ GF2(result)), syndrome_1d), "decoded result does not reproduce the syndrome"
            assert result.any(), "decoder returned all-zero (no operator found)"

    result=result.reshape(1, -1)

    return sc.turn_TXZYerror_into_tableau(result)

def find_short_second_logical(T,first_logical,CSS:bool=False,max_iter:int=100,p_base:float=0.1,p_punish:float=0.01):

    #finds second logical with hopefully small overlap

    if sc.rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')

    n_qubits=T.shape[1]//2 #integer division

    T=sc.to_gf2_tableau(T)

    if isinstance(first_logical,list):
        first_logical=sc.tableau_list_to_matrix(first_logical)

    if CSS:
        pass

    T=sc.append_logical_to_tableau(T,first_logical)
    T=sc.turn_tableau_into_TXZY(T)

    syndrome = np.zeros(T.shape[0], dtype=int)
    syndrome[-1]=1 #assumes logical qubit is last row
    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)


    if CSS:
        pass

    else:
        channel_probs = create_error_probs_from_logical(n_qubits,first_logical,p_base=p_base,p_punish=p_punish)

        decoder = BpOsdDecoder(T, channel_probs=channel_probs,
                                max_iter=max_iter, bp_method="ms",
                                osd_method="osd_cs", osd_order=6)

        result = decoder.decode(syndrome_1d)

        if result is None or not np.asarray(result).any():
            return None

        else:

            assert np.array_equal((T @ GF2(result)), syndrome_1d), "decoded result does not reproduce the syndrome"
            assert result.any(), "decoder returned all-zero (no operator found)"

    result=result.reshape(1, -1)

    return sc.turn_TXZYerror_into_tableau(result)


def create_graph_code(in_adj : np.array, code_node : int = 0):
    """create a graph code from an input graph
    using the 0th node as the input node"""

    num_nodes = in_adj.shape[0]
    identity = np.identity(num_nodes, dtype = np.uint16)
    gen = np.hstack([identity, in_adj])
    zlogi = gen[code_node].copy()
    zlogi[code_node] = 0
    neigh = [i for i, ele in enumerate(in_adj[code_node]) if ele ==1]

    assert(len(neigh)>0)
    xlogi = gen[neigh[0]].copy()

    for ele in neigh[1:]:
        gen[ele] ^= gen[neigh[0]]

    rows_to_remove = [code_node, neigh[0]]
    gen = np.delete(gen, rows_to_remove, axis=0)

    gen = np.delete(gen, [code_node, code_node+num_nodes], axis=1)
    xlogi = np.delete(xlogi, [code_node, code_node+num_nodes])#, axis=1)
    zlogi = np.delete(zlogi, [code_node, code_node+num_nodes])#, axis=1)


    return xlogi, zlogi, gen


def main():

    numq = 400
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    print('xlogical input: ',tableau2paulistring(xlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=sc.tableau_list_to_matrix(stabi)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lost_qubits=[0,1]
    short_z=find_short_first_logical(T,xlogi)
    #x=find_short_second_logical(T,short_z)
    nonzero=np.flatnonzero(short_z.ravel())
    #print('nonzero indices', nonzero)
    print('short zlogical from decoder: ',tableau2paulistring(short_z))
    #print('xlogical decoder',tableau2paulistring(x))
    short_x=sc.find_anti_commuting_logi_at_O(T,short_z,nonzero[0])
    print('xlogical anticommuting at {}, linear algebra method: '.format(nonzero[0]),(tableau2paulistring(short_x)))

    return short_z




if __name__ == "__main__":

    main()



