import numpy as np
import stabiliser_code as sc
from ldpc import BpOsdDecoder
from galois import GF2
import networkx as nx
from tableau import tableau2paulistring
from stabiliser_code import create_graph_code
import tableau as ta



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

def make_given_logical_unlikely(channel_probs, num_qubits, logical, p_punish: float = 0.2):
    """
    Punish the decoder channel on the TXZY support of a found solution vector logical
    (x|z repr), so BP+OSD avoids returning that same vector again. In place.
    """
    logical = ta.to_gf2_tableau(logical)

    if logical.ndim == 2 and logical.shape[1] // 2 != num_qubits:
        raise ValueError(f"Logical on {logical.shape[1]//2} qubits, but num_qubits is {num_qubits}")
    if logical.ndim == 1 and logical.shape[0] // 2 != num_qubits:
        raise ValueError(f"Logical on {logical.shape[0]//2} qubits, but num_qubits is {num_qubits}")

    assert channel_probs.ravel().shape[0] == 3 * num_qubits, \
        f"channel_probs must have {3*num_qubits} entries"

    logical_txzy = ta.turn_tableau_into_TXZY(logical)     
    nonzero = np.asarray(logical_txzy).ravel().astype(bool)

    channel_probs[nonzero] = p_punish

    return channel_probs

def bias_channel_toward_qubit(channel_probs, o, n_qubits, p_low=0.2, p_high=0.3):
    """outputs channel probabilities that make it likely that an error happend on qubit o"""
    channel_probs = np.asarray(channel_probs, dtype=float).copy()
    rng = np.random.default_rng()
    channel_probs[o]              = rng.uniform(p_low, p_high)  # X at O
    channel_probs[o + n_qubits]   = rng.uniform(p_low, p_high)  # Z at O


    channel_probs[o + 2*n_qubits] =0  # Y at O
    #TODO: figure out how (3n) format maps to probabilities and how to best bias towards solution
    #with support on o
    return channel_probs



def find_first_short_logical_in_coset(T, logi_commute, logi_anticommute,
                                       remaining_logicals:list=None, o=None,
                                       max_iter=100, channel_probs=None):
    
    """Finds a short logical of a tableau T that is in the same logical coset as logi_commute and anti-
    commutes with logi_anticommute. Option to add remaining logicals as argument, acting on the other logical qubits 
    than logi_commute and logi_anticommute. Outputs a first_short_logical that COMMUTES with all remaining logicals. 
    Remaining logicals must be a SYMPLECTIC BASIS"""

    if sc.rank_F2(T) < T.shape[0]:
        raise ValueError('T not full rank')

    n_qubits = T.shape[1] // 2
    T = ta.to_gf2_tableau(T)

    logi_anticommute = ta.to_gf2_tableau(logi_anticommute)
    logi_commute = ta.to_gf2_tableau(logi_commute)
    remaining_logicals = [ta.to_gf2_tableau(r) for r in remaining_logicals]


     
    if channel_probs is None:
        channel_probs = np.full(3 * n_qubits, 0.1)

    if o is not None:
        channel_probs = bias_channel_toward_qubit(channel_probs, o, n_qubits) #obtain channel_probs that
        #bias qubit support on qubit o

    T_aug = T
    print('T_aug shape',T_aug.shape)
    for r in remaining_logicals:
         
        T_aug = sc.append_logical_to_tableau(T_aug, r)

    T_aug = sc.append_logical_to_tableau(T_aug, logi_commute)
    T_aug = sc.append_logical_to_tableau(T_aug, logi_anticommute)
    T_aug = ta.turn_tableau_into_TXZY(T_aug)

    r = sc.rank_F2(T_aug)
    if r < T_aug.shape[0]:
        print(f"T_aug rank deficient after appends: {r} < {T_aug.shape[0]}")

    syndrome = np.zeros(T_aug.shape[0], dtype=int)
    syndrome[-1] = 1
    syndrome_1d = syndrome.astype(np.uint8)
    channel_probs = np.full(3 * n_qubits, 0.1) #test 
    decoder = BpOsdDecoder(T_aug, channel_probs=channel_probs,
                            max_iter=max_iter, bp_method="ms",
                            osd_method="osd_cs", osd_order=6)
    result = decoder.decode(syndrome_1d)

    if result is None or not np.asarray(result).any():
        return None, channel_probs

    actual = np.asarray(T_aug @ GF2(result.reshape(-1, 1))).ravel()
    if not np.array_equal(np.asarray(actual).astype(int), np.asarray(syndrome_1d).astype(int)):
        return None, channel_probs

    result_tab = ta.turn_TXZYerror_into_tableau(result.reshape(1, -1))
 
    v = np.asarray(result_tab).ravel().astype(int)
    

    if o is not None and not (v[o] or v[o + n_qubits]):
        return None, channel_probs

    return ta.to_gf2_tableau(v.reshape(1, -1)), channel_probs
 

def find_short_first_logical(T,other_logical_repr,CSS:bool=False,max_iter:int=100,channel_probs=None): #TODO: what happens if result is None? This may happen?


    #decoder problem now takes on representation (x|z|x plus z)
    #when giving decoder matrix in (x|z|x+z) format and without Omega, the error that is given back is in
    #(z|x|x+z) format!
    #TODO

    if sc.rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')

    n_qubits=T.shape[1]//2 #integer division

    T=ta.to_gf2_tableau(T)

    if isinstance(other_logical_repr,list):
        other_logical_repr=sc.tableau_list_to_matrix(other_logical_repr)

    if CSS:
        print("css")
        pass

    T=sc.append_logical_to_tableau(T,other_logical_repr)
    T=ta.turn_tableau_into_TXZY(T)

    syndrome = np.zeros(T.shape[0], dtype=int)
    syndrome[-1]=1 #assumes logical qubit is last row
    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)


    if CSS:
        pass

    else:
        if channel_probs is None:
            channel_probs = np.full(3 * n_qubits, 0.1)

        decoder = BpOsdDecoder(T, channel_probs=channel_probs,
                                max_iter=max_iter, bp_method="ms",
                                osd_method="osd_cs", osd_order=6)
        result = decoder.decode(syndrome_1d)

        if result is None or not np.asarray(result).any():
            return None,channel_probs
        else:
            assert np.array_equal((T @ GF2(result)), syndrome_1d), "decoded result does not reproduce the syndrome"
            assert result.any(), "decoder returned all-zero (no operator found)"

    result=result.reshape(1, -1)

    return ta.turn_TXZYerror_into_tableau(result),channel_probs


def find_short_second_logical(T,first_logical,CSS:bool=False,\
                              max_iter:int=100,p_base:float=0.1,p_punish:float=0.01):

    #finds second logical with hopefully small overlap

    if sc.rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')

    n_qubits=T.shape[1]//2 #integer division

    T=ta.to_gf2_tableau(T)

    if isinstance(first_logical,list):
        first_logical=sc.tableau_list_to_matrix(first_logical)

    if CSS:
        pass

    T=sc.append_logical_to_tableau(T,first_logical)
    T=ta.turn_tableau_into_TXZY(T)

    syndrome = np.zeros(T.shape[0], dtype=int)
    syndrome[-1]=1 #assumes logical qubit is last row
    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)


    if CSS:
        pass

    else:
        
        channel_probs =create_error_probs_from_logical(n_qubits,first_logical)
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

    return ta.turn_TXZYerror_into_tableau(result)



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



